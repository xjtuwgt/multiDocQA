# import torch
# torch.manual_seed(43)
from hotpotEvaluation.hotpotEvaluationUtils import answer_span_evaluation_in_sentence
#
# x = torch.rand(100)
# y = torch.rand(100)
#
# print(x.shape[0], y.shape[0])
#
# print(x)
# print(y)
#
# z = answer_span_evaluation_in_sentence(start_scores=x, end_scores=y, max_ans_decode_len=20, debug=True)
# print(z)

from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
encoding = tokenizer(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]

outputs = model(input_ids, attention_mask=attention_mask)
print(outputs)
start_logits = outputs[0]
end_logits = outputs[1]
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

print(all_tokens)

print(start_logits)
print(end_logits)

answer = answer_span_evaluation_in_sentence(start_scores=torch.sigmoid(start_logits[0]), end_scores=torch.sigmoid(end_logits[0]))

print(all_tokens[answer[1]:(answer[2]+1)])

print(answer)

answer_tokens = all_tokens[torch.argmax(start_logits) :torch.argmax(end_logits)+1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)) # remove space prepending space token
print(answer)