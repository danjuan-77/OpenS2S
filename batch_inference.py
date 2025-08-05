#!/usr/bin/env python3
"""
OpenS2S批量推理脚本
基于inference.py，支持处理JSONL文件中的音频数据批量推理
"""
import os
# 设置环境变量，强制使用soundfile后端
os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'
import torch
import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
    print("使用soundfile音频后端")
except Exception as e:
    print(f"警告：无法设置soundfile后端，将使用默认后端: {e}")
import argparse
import os
import sys
import tempfile
import base64
import uuid
import json
from transformers import AutoTokenizer, GenerationConfig
from copy import deepcopy
from tqdm import tqdm

# 添加路径（与inference.py保持一致）
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
class OpenS2SBatchInference:
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
        
        # 加载音频特征提取器
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
        
        # 设置units bias
        self.units_bias = self.tts_tokenizer.encode("<|audio_0|>")[0]
    
    def get_input_params(self, messages):
        """处理消息输入，与inference.py中的方法保持一致"""
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
    
    def _prepare_input_from_wav(self, wav_path, text_prompt=""):
        """从音频文件路径准备输入"""
        messages = []
        
        # 读取音频文件并转换为base64
        with open(wav_path, 'rb') as f:
            audio_binary = f.read()
        audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
        
        if text_prompt:
            content = [{"audio": audio_base64}, {"text": text_prompt}]
        else:
            content = {"audio": audio_base64}
        
        messages.append({"role": "user", "content": content})
        return self.get_input_params(messages)
    
    @torch.inference_mode()
    def inference_single(self, wav_path, text_prompt="", output_path=None, temperature=1.0, top_p=1.0, max_new_tokens=256):
        """单个音频文件推理"""
        # 准备输入
        input_ids, speech_values, speech_mask = self._prepare_input_from_wav(wav_path, text_prompt)
        input_ids = input_ids.to(device=self.device, non_blocking=True)
        
        if speech_values is not None:
            speech_values = speech_values.to(dtype=torch.bfloat16, device=self.device, non_blocking=True)
            speech_mask = speech_mask.to(device=self.device, non_blocking=True)
        
        # 设置生成参数
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
        
        # 执行生成
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
        
        # 处理输出
        input_length = len(input_ids[0])
        
        # 处理模型输出
        if isinstance(outputs, dict):
            text_outputs = outputs.get('sequences')
            units_outputs = outputs.get('units')
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            text_outputs, units_outputs = outputs
        elif hasattr(outputs, 'sequences'):
            text_outputs = outputs.sequences
            units_outputs = getattr(outputs, 'units_sequences', None)
        else:
            text_outputs = outputs
            units_outputs = None
        
        # 确保 text_outputs 是tensor
        if not torch.is_tensor(text_outputs):
            raise ValueError(f"Unexpected text_outputs type: {type(text_outputs)}")
        
        # 解码文本响应
        if len(text_outputs.shape) > 1:
            response_text = self.tokenizer.decode(
                text_outputs[0][input_length:], 
                skip_special_tokens=True
            )
        else:
            response_text = self.tokenizer.decode(
                text_outputs[input_length:], 
                skip_special_tokens=True
            )
        
        # 生成音频
        audio_saved = False
        if output_path and units_outputs is not None and torch.is_tensor(units_outputs):
            try:
                # 提取units并减去bias
                if len(units_outputs.shape) > 1:
                    generated_units = units_outputs[0][input_length:]
                else:
                    generated_units = units_outputs[input_length:]
                
                units = []
                for unit_id in generated_units:
                    unit_value = unit_id.item() - self.units_bias
                    if unit_value >= 0:
                        units.append(unit_value)
                
                if units:
                    # 使用AudioDecoder生成音频
                    session_id = uuid.uuid4()
                    tts_token = torch.LongTensor(units).unsqueeze(0).to(device=self.device)
                    
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
                    torchaudio.save(output_path, audio_output.cpu(), 22050)
                    audio_saved = True
                    
            except Exception as e:
                print(f"音频生成失败: {e}")
        
        return {
            "text": response_text,
            "audio_saved": audio_saved
        }

def main():
    parser = argparse.ArgumentParser(description="OpenS2S Infer")
    parser.add_argument("--model-path", type=str, default="/share/nlp/tuwenming/models/CASIA-LM/OpenS2S", 
                       help="OpenS2S模型路径")
    parser.add_argument("--flow-path", type=str, default="/share/nlp/tuwenming/models/zai-org/glm-4-voice-decoder", 
                       help="Flow解码器模型路径")
    parser.add_argument("--test-file", type=str, default="/share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl", 
                       help="测试JSONL文件路径")
    parser.add_argument("--wav-prefix", type=str, default="/share/nlp/tuwenming/projects/UltraVoice_dev/data", 
                       help="音频文件路径前缀")
    parser.add_argument("--save-dir", type=str, required=True, 
                       help="输出音频保存目录")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"], help="使用的设备")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="生成温度参数")
    parser.add_argument("--top-p", type=float, default=0.8,
                       help="nucleus sampling参数")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="最大生成token数量")
    
    # 数据范围参数
    parser.add_argument("--start", type=int, default=0,
                       help="开始处理的数据行号（从0开始）")
    parser.add_argument("--end", type=int, default=-1,
                       help="结束处理的数据行号（不包含该行，-1表示处理到文件末尾）")
    
    args = parser.parse_args()
    
    # 检查输入
    if not os.path.exists(args.model_path):
        print(f"错误：模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.flow_path):
        print(f"错误：Flow模型路径不存在: {args.flow_path}")
        return
    
    if not os.path.exists(args.test_file):
        print(f"错误：测试文件不存在: {args.test_file}")
        return
    
    if not os.path.exists(args.wav_prefix):
        print(f"错误：音频前缀路径不存在: {args.wav_prefix}")
        return
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # 初始化推理器
        print(f"使用设备: {args.device}")
        print(f"模型路径: {args.model_path}")
        print(f"Flow路径: {args.flow_path}")
        print(f"测试文件: {args.test_file}")
        print(f"音频前缀: {args.wav_prefix}")
        print(f"保存目录: {args.save_dir}")
        
        inferencer = OpenS2SBatchInference(args.model_path, args.flow_path, args.device)
        
        # 统计信息
        total_count = 0
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        # 首先统计总行数
        with open(args.test_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # 确定处理范围
        start_idx = max(0, args.start)
        end_idx = min(total_lines, args.end) if args.end > 0 else total_lines
        
        if start_idx >= end_idx:
            print(f"错误：无效的数据范围 [{start_idx}, {end_idx})，总数据行数：{total_lines}")
            return
        
        process_lines = end_idx - start_idx
        print(f"\n开始批量推理，总共 {total_lines} 条数据，处理范围 [{start_idx}, {end_idx})，共 {process_lines} 条数据...")
        
        # 处理JSONL文件
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # 跳过不在处理范围内的数据
                if line_num < start_idx:
                    continue
                if line_num >= end_idx:
                    break
                
                # 使用相对行号显示进度
                relative_line_num = line_num - start_idx + 1
                if relative_line_num == 1:
                    # 初始化进度条
                    progress_bar = tqdm(total=process_lines, desc="处理进度")
                
                progress_bar.update(1)
                try:
                    total_count += 1
                    data = json.loads(line.strip())
                    
                    # 提取音频路径
                    wav_path = data['instruction_wav_path']
                    full_wav_path = os.path.join(args.wav_prefix, wav_path)
                    
                    # 生成输出文件名
                    output_filename = f"{data['split_type']}_{data['sub_type']}_{data['data_source']}_{data['index']}.wav"
                    output_path = os.path.join(args.save_dir, output_filename)
                    
                    # 检查输出文件是否已存在
                    if os.path.exists(output_path):
                        print(f"跳过第{line_num}行(相对第{relative_line_num}行): 输出文件已存在: {output_path}")
                        skipped_count += 1
                        continue
                    
                    # 检查输入音频文件是否存在
                    if not os.path.exists(full_wav_path):
                        print(f"跳过第{line_num}行(相对第{relative_line_num}行): 输入音频文件不存在: {full_wav_path}")
                        failed_count += 1
                        continue
                    
                    # 执行推理
                    text_prompt = data.get('instruction', '')  # 如果有指令文本
                    results = inferencer.inference_single(
                        wav_path=full_wav_path,
                        text_prompt=text_prompt,
                        output_path=output_path,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens
                    )
                    
                    # 打印文本结果（不保存）
                    print(f"第{line_num}行(相对第{relative_line_num}行)推理完成:")
                    print(f"  输入: {full_wav_path}")
                    print(f"  文本响应: {results['text']}")
                    if results['audio_saved']:
                        print(f"  音频已保存: {output_path}")
                    else:
                        print("  音频保存失败")
                    print("-" * 50)
                    
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行(相对第{relative_line_num}行)JSON解析错误: {e}")
                    failed_count += 1
                    continue
                except KeyError as e:
                    print(f"第{line_num}行(相对第{relative_line_num}行)缺少必要字段: {e}")
                    failed_count += 1
                    continue
                except Exception as e:
                    print(f"第{line_num}行(相对第{relative_line_num}行)推理失败: {e}")
                    failed_count += 1
                    continue
        
        # 关闭进度条
        if 'progress_bar' in locals():
            progress_bar.close()
        
        # 输出统计信息
        print("\n=== 批量推理完成 ===")
        print(f"数据范围: [{start_idx}, {end_idx})")
        print(f"总计: {total_count} 条数据")
        print(f"成功处理: {processed_count} 条")
        print(f"跳过（已存在）: {skipped_count} 条")
        print(f"失败: {failed_count} 条")
        print(f"输出目录: {args.save_dir}")
        
    except Exception as e:
        print(f"批量推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
