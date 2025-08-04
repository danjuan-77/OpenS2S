#!/usr/bin/env python3
"""
OpenS2S简单推理脚本
支持音频和文本输入，输出文本和音频响应
"""

import torch
import torchaudio
import argparse
import os
import sys
import tempfile
import base64
import uuid
from io import BytesIO
from transformers import AutoTokenizer, GenerationConfig
from threading import Thread
import json

# 导入OpenS2S相关模块
from src.modeling_omnispeech import OmniSpeechModel
from src.feature_extraction_audio import WhisperFeatureExtractor
from src.utils import get_waveform
from flow_inference import AudioDecoder
from src.constants import (
    DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, 
    DEFAULT_AUDIO_TOKEN, DEFAULT_TTS_START_TOKEN, AUDIO_TOKEN_INDEX
)

class OpenS2SInference:
    def __init__(self, model_path, flow_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        self.flow_path = flow_path
        
        print("正在加载模型...")
        self._load_models()
        print("模型加载完成！")
    
    def _load_models(self):
        """加载所有必要的模型组件"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # 加载主模型
        self.model = OmniSpeechModel.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.to(self.device).eval()
        
        # 加载音频特征提取器
        self.audio_extractor = WhisperFeatureExtractor.from_pretrained(self.model_path)
        
        # 加载音频解码器
        config_path = os.path.join(self.flow_path, "config.yaml")
        flow_ckpt_path = os.path.join(self.flow_path, "flow.pt")
        hift_ckpt_path = os.path.join(self.flow_path, "hift.pt")
        
        self.audio_decoder = AudioDecoder(
            config_path, flow_ckpt_path, hift_ckpt_path, device=self.device
        )
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def _prepare_input(self, audio_path=None, text_prompt=""):
        """准备模型输入"""
        audios = []
        
        # 处理音频输入
        if audio_path and os.path.exists(audio_path):
            waveform = get_waveform(audio_path)
            audios.append(waveform)
        
        # 构建消息
        if audios:
            # 有音频输入
            content = f"{text_prompt}{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
        else:
            # 只有文本输入
            content = text_prompt
        
        messages = [{"role": "user", "content": content}]
        
        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False, 
            enable_thinking=False
        )
        prompt += DEFAULT_TTS_START_TOKEN
        
        # 分词处理
        segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids += [AUDIO_TOKEN_INDEX]
            input_ids += self.tokenizer.encode(segment)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        
        # 处理音频特征
        if audios:
            speech_inputs = self.audio_extractor(
                audios,
                sampling_rate=self.audio_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features
            speech_mask = speech_inputs.attention_mask
        else:
            speech_values, speech_mask = None, None
        
        return input_ids, speech_values, speech_mask
    
    def _extract_speech_units(self, output_ids, input_length):
        """从输出中提取speech units（简化版本）"""
        # 这是一个简化的实现，实际的speech units提取会更复杂
        generated_ids = output_ids[0][input_length:]
        
        # 查找TTS相关的token
        speech_units = []
        for token_id in generated_ids:
            # 这里需要根据实际的token mapping来处理
            # 这只是一个占位符实现
            if token_id != self.tokenizer.eos_token_id:
                speech_units.append(token_id.item())
        
        return torch.tensor(speech_units).unsqueeze(0) if speech_units else None
    
    def inference(self, audio_path=None, text_prompt="", output_dir="./output"):
        """执行推理"""
        if not text_prompt and not audio_path:
            raise ValueError("请提供文本提示或音频文件")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备输入
        input_ids, speech_values, speech_mask = self._prepare_input(audio_path, text_prompt)
        input_ids = input_ids.to(self.device)
        
        if speech_values is not None:
            speech_values = speech_values.to(dtype=torch.bfloat16, device=self.device)
            speech_mask = speech_mask.to(self.device)
        
        print("正在生成响应...")
        
        # 执行生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=None,
                speech_values=speech_values,
                speech_mask=speech_mask,
                spk_emb=None,
                units_gen=True,
                generation_config=self.generation_config
            )
        
        # 解码文本响应
        input_length = len(input_ids[0])
        response_text = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        # 保存文本输出
        text_output_path = os.path.join(output_dir, "response.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        print(f"文本响应: {response_text}")
        print(f"文本已保存到: {text_output_path}")
        
        # 尝试生成音频（简化版本）
        try:
            # 提取speech units
            speech_units = self._extract_speech_units(outputs, input_length)
            
            if speech_units is not None and len(speech_units[0]) > 0:
                # 使用AudioDecoder生成音频
                session_id = str(uuid.uuid4())
                audio_output = self.audio_decoder.token2wav(
                    speech_units.to(self.device),
                    session_id,
                    finalize=True
                )
                
                # 保存音频输出
                audio_output_path = os.path.join(output_dir, "response.wav")
                torchaudio.save(audio_output_path, audio_output, 22050)
                print(f"音频已保存到: {audio_output_path}")
            else:
                print("未生成音频输出")
                
        except Exception as e:
            print(f"音频生成出错: {e}")
            print("只有文本输出可用")
        
        return {
            "text": response_text,
            "text_file": text_output_path,
            "audio_file": os.path.join(output_dir, "response.wav") if 'audio_output_path' in locals() else None
        }

def main():
    parser = argparse.ArgumentParser(description="OpenS2S推理脚本")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="OpenS2S模型路径")
    parser.add_argument("--flow-path", type=str, required=True, 
                       help="Flow解码器模型路径")
    parser.add_argument("--audio-path", type=str, default="", 
                       help="输入音频文件路径")
    parser.add_argument("--text", type=str, default="", 
                       help="输入文本提示")
    parser.add_argument("--output-dir", type=str, default="./output", 
                       help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"], help="使用的设备")
    
    args = parser.parse_args()
    
    # 检查输入
    if not args.text and not args.audio_path:
        print("错误：请提供 --text 或 --audio-path 参数")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.flow_path):
        print(f"错误：Flow模型路径不存在: {args.flow_path}")
        return
    
    if args.audio_path and not os.path.exists(args.audio_path):
        print(f"错误：音频文件不存在: {args.audio_path}")
        return
    
    try:
        # 初始化推理器
        inferencer = OpenS2SInference(args.model_path, args.flow_path, args.device)
        
        # 执行推理
        results = inferencer.inference(
            audio_path=args.audio_path if args.audio_path else None,
            text_prompt=args.text,
            output_dir=args.output_dir
        )
        
        print("\n推理完成！")
        print(f"输出目录: {args.output_dir}")
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()