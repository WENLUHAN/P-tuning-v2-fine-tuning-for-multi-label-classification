# 微调后
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

model_path = "chatglm2-6b-4int"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 微调后代码
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
# 加载根据ADGEN数据集微调后训练的模型
prefix_state_dict = torch.load(os.path.join("/home/featurize/work/ChatGLM2-6B/ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-3000", "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

prompt = "分类任务：为以下文本匹配相关的标签。标签只能从以下20个标签中选择：长期合作，时限性，还款灵活，不是催借款，活动区别，提前还款政策，利息高，无隐性费用，确认理解，活动真实有效，稀缺性，不建议提前还款，放款快，非强制参与，回拨联系，对比利息，循环借款，人工登记，无标签，成功邀请。\n输出指南：只需要输出匹配的标签，无需过多解释，多个匹配项用'&'连接。例如：时限性&放款快\n文本：嗯您好先生，我这边呢是您在合作使用的三六零借条的工号八六八九，先生，我们来电呢也是回馈老客户呢，针对部分资质比较好的客户呢，给您下发了一个三至十五天的周转金免息券的，如果说您这边领到这个周转金免息券的话，在这个。"
# 模型输出
current_length = 0
for response, history in model.stream_chat(tokenizer, prompt, history=[]):
    print(response[current_length:], end="", flush=True)
    current_length = len(response)
print("")
