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
from transformers import AutoTokenizer, GenerationConfig
from copy import deepcopy

# 添加路径（与model_worker.py保持一致）
sys.path.append("cosyvoice")
sys.path.append("third_party/Matcha-TTS")

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
        self.system_prompt = "You are a helpful assistant."
        
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
        
        # 加载TTS tokenizer
        self.tts_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.model_path, "tts"),
            trust_remote_code=True
        )
        
        # 加载主模型
        self.model = OmniSpeechModel.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.to(self.device).eval()
        
        # 加载音频特征提取器 - 关键修复：从audio子目录加载
        self.audio_extractor = WhisperFeatureExtractor.from_pretrained(
            os.path.join(self.model_path, "audio")
        )
        
        # 加载音频解码器
        config_path = os.path.join(self.flow_path, "config.yaml")
        flow_ckpt_path = os.path.join(self.flow_path, "flow.pt")
        hift_ckpt_path = os.path.join(self.flow_path, "hift.pt")
        
        self.audio_decoder = AudioDecoder(
            config_path, flow_ckpt_path, hift_ckpt_path, device=self.device
        )
        
        # 设置生成配置
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.tts_generation_config = GenerationConfig.from_pretrained(
            os.path.join(self.model_path, "tts")
        )
        
        # 设置units bias（用于TTS token处理）
        self.units_bias = self.tts_tokenizer.encode("<|audio_0|>")[0]
    
    def get_input_params(self, messages):
        """处理消息输入，与model_worker.py中的方法保持一致"""
        new_messages = []
        audios = []
        if self.system_prompt:
            new_messages.append({"role": "system", "content": self.system_prompt})
        
        for turn in messages:
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                new_content = ""
                for item in content:
                    if item.get("audio", ""):
                        audio_binary = base64.b64decode(item["audio"])
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                            temp_file.write(audio_binary)
                            temp_file_path = temp_file.name
                            waveform = get_waveform(temp_file_path)
                            audios.append(waveform)
                        new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                    elif item.get("text", ""):
                        new_content += item["text"]
            elif isinstance(content, dict):
                new_content = ""
                if content.get("audio", ""):
                    audio_binary = base64.b64decode(content["audio"])
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                        temp_file.write(audio_binary)
                        temp_file_path = temp_file.name
                        waveform = get_waveform(temp_file_path)
                        audios.append(waveform)
                    new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                elif content.get("text", ""):
                    new_content += content["text"]
            else:
                raise NotImplementedError
            new_messages.append({"role": f"{role}", "content": f"{new_content}"})

        prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompt += DEFAULT_TTS_START_TOKEN
        segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids += [AUDIO_TOKEN_INDEX]
            input_ids += self.tokenizer.encode(segment)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)

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
    
    def _prepare_input(self, audio_path=None, text_prompt=""):
        """简化的输入准备方法，将音频文件和文本转换为消息格式"""
        messages = []
        
        # 构建content
        if audio_path and os.path.exists(audio_path):
            # 读取音频文件并转换为base64
            with open(audio_path, 'rb') as f:
                audio_binary = f.read()
            audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
            content = {"audio": audio_base64}
            if text_prompt:
                content = [{"audio": audio_base64}, {"text": text_prompt}]
        else:
            content = text_prompt if text_prompt else "你好"
        
        messages.append({"role": "user", "content": content})
        return self.get_input_params(messages)
    

    @torch.inference_mode()
    def inference(self, audio_path=None, text_prompt="", output_dir="./output", temperature=1.0, top_p=1.0, max_new_tokens=256):
        """执行推理，基于model_worker.py的生成逻辑"""
        if not text_prompt and not audio_path:
            raise ValueError("请提供文本提示或音频文件")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备输入
        input_ids, speech_values, speech_mask = self._prepare_input(audio_path, text_prompt)
        input_ids = input_ids.to(device=self.device, non_blocking=True)
        
        if speech_values is not None:
            speech_values = speech_values.to(dtype=torch.bfloat16, device=self.device, non_blocking=True)
            speech_mask = speech_mask.to(device=self.device, non_blocking=True)
        
        print("正在生成响应...")
        
        # 设置生成参数（与model_worker.py保持一致）
        generation_config = deepcopy(self.generation_config)
        tts_generation_config = deepcopy(self.tts_generation_config)
        
        do_sample = True if temperature > 0.001 else False
        max_new_tokens = min(int(max_new_tokens), 1024)
        
        generation_config.update(
            **{
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        
        tts_generation_config.update(
            **{
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0
            }
        )
        
        # 执行生成（非流式版本）
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=None,
            speech_values=speech_values,
            speech_mask=speech_mask,
            spk_emb=None,
            units_gen=True,
            generation_config=generation_config,
            tts_generation_config=tts_generation_config,
            use_cache=True,
        )
        
        # 处理输出（基于model_worker.py的逻辑）
        input_length = len(input_ids[0])
        
        # 处理模型输出（基于实际返回的字典格式）
        if isinstance(outputs, dict):
            text_outputs = outputs.get('sequences')
            units_outputs = outputs.get('units')
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            text_outputs, units_outputs = outputs
        elif hasattr(outputs, 'sequences'):
            # GenerateOutput 对象
            text_outputs = outputs.sequences
            units_outputs = getattr(outputs, 'units_sequences', None)
        else:
            # 直接tensor输出
            text_outputs = outputs
            units_outputs = None
        
        # 确保 text_outputs 是tensor
        if not torch.is_tensor(text_outputs):
            raise ValueError(f"Unexpected text_outputs type: {type(text_outputs)}")
        
        # 解码文本响应
        if len(text_outputs.shape) > 1:
            # 批处理输出
            response_text = self.tokenizer.decode(
                text_outputs[0][input_length:], 
                skip_special_tokens=True
            )
        else:
            # 单个序列输出
            response_text = self.tokenizer.decode(
                text_outputs[input_length:], 
                skip_special_tokens=True
            )
        
        # 保存文本输出
        text_output_path = os.path.join(output_dir, "response.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        print(f"文本响应: {response_text}")
        print(f"文本已保存到: {text_output_path}")
        
        # 生成音频（基于model_worker.py的音频生成逻辑）
        audio_output_path = None
        try:
            if units_outputs is not None and torch.is_tensor(units_outputs):
                # 提取units并减去bias
                if len(units_outputs.shape) > 1:
                    generated_units = units_outputs[0][input_length:]
                else:
                    generated_units = units_outputs[input_length:]
                
                units = []
                for unit_id in generated_units:
                    unit_value = unit_id.item() - self.units_bias
                    if unit_value >= 0:  # 只保留有效的audio units
                        units.append(unit_value)
                
                print(f"提取到 {len(units)} 个有效的音频units")
                
                if units:
                    # 使用AudioDecoder生成音频
                    session_id = uuid.uuid4()
                    tts_token = torch.LongTensor(units).unsqueeze(0).to(device=self.device)
                    
                    # 生成音频（使用与model_worker相同的参数）
                    prompt_speech_feat = torch.zeros(1, 0, 80).to(device=self.device)
                    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device=self.device)
                    
                    audio_output, _ = self.audio_decoder.token2wav(
                        tts_token, 
                        uuid=session_id,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        finalize=True
                    )
                    
                    # 保存音频输出
                    audio_output_path = os.path.join(output_dir, "response.wav")
                    torchaudio.save(audio_output_path, audio_output.cpu(), 22050)
                    print(f"音频已保存到: {audio_output_path}")
                else:
                    print("未生成有效的音频units")
            else:
                print("模型未返回音频units")
                
        except Exception as e:
            print(f"音频生成出错: {e}")
            import traceback
            traceback.print_exc()
            print("只有文本输出可用")
        
        return {
            "text": response_text,
            "text_file": text_output_path,
            "audio_file": audio_output_path
        }

def main():
    parser = argparse.ArgumentParser(description="OpenS2S推理脚本（基于model_worker.py）")
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
    
    # 生成参数（与model_worker.py保持一致）
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="生成温度参数")
    parser.add_argument("--top-p", type=float, default=0.8,
                       help="nucleus sampling参数")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="最大生成token数量")
    
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
        print(f"使用设备: {args.device}")
        print(f"模型路径: {args.model_path}")
        print(f"Flow路径: {args.flow_path}")
        
        inferencer = OpenS2SInference(args.model_path, args.flow_path, args.device)
        
        # 执行推理
        results = inferencer.inference(
            audio_path=args.audio_path if args.audio_path else None,
            text_prompt=args.text,
            output_dir=args.output_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens
        )
        
        print("\n=== 推理完成 ===")
        print(f"输出目录: {args.output_dir}")
        print(f"文本响应: {results['text']}")
        if results['audio_file']:
            print(f"音频文件: {results['audio_file']}")
        else:
            print("未生成音频文件")
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()