import torch
import torch.nn as nn
from transformers import BertModel


class BlockTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BlockTransformer, self).__init__()
        self.transformer = BertModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size

    def forward(self, input_ids, attention_mask, past_key_values=None):
        # Transformer的前向计算
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   past_key_values=past_key_values,
                                   use_cache=True)
        # 提取输出和新的缓存值
        hidden_states = outputs.last_hidden_state
        new_past_key_values = outputs.past_key_values
        return hidden_states, new_past_key_values


def process_streaming_input(block_transformer, input_stream, block_size=100):
    past_key_values = None
    all_outputs = []

    for i in range(0, len(input_stream), block_size):
        # 分块处理输入
        input_ids = input_stream[i:i + block_size]
        attention_mask = torch.ones_like(input_ids)  # 假设所有输入都是有效的

        # 将当前块和之前的上下文信息一起传入模型
        hidden_states, past_key_values = block_transformer(input_ids, attention_mask, past_key_values)

        # 保存当前块的输出
        all_outputs.append(hidden_states)

    # 最终输出结果
    return torch.cat(all_outputs, dim=1)


# 模拟输入数据
input_stream = torch.randint(0, 1000, (1, 500))  # 假设输入序列长度为500
block_size = 100

# 初始化模型
block_transformer = BlockTransformer()

# 处理流式输入
output = process_streaming_input(block_transformer, input_stream, block_size)
print(output.shape)  # 输出的形状应为 (batch_size, sequence_length, hidden_size)
