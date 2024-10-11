import os
import torch
import pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer

CHECKPOINT_PATH = "/home/featurize/work/ChatGLM2-6B/ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-6000"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("chatglm2-6b-4int", trust_remote_code=True)

config = AutoConfig.from_pretrained("chatglm2-6b-4int", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("chatglm2-6b-4int", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

final_data = []
test_data_path = "data_test.xlsx"
test_data = pd.read_excel(test_data_path)

for i in range(len(test_data)):
    print(f"正在预测第{i+1}条文本")
    text = test_data.loc[i, "text"]
    label = test_data.loc[i, "label"]
    response, history = model.chat(tokenizer, text, history=[])

    final_data.append({
        "text": text,
        "label": label,
        "prediction": response
    })
df = pd.DataFrame(final_data)
df.to_excel("prediction.xlsx", index=False)
