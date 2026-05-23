# nanochat 预训练学习笔记

这份文档记录从 Node.js 技术栈转向深度学习训练时，围绕 nanochat 预训练流程的关键问题。

## 问题 1：预训练阶段只需要 `text` 就可以了吗？

是的，base model 预训练阶段只需要大量 `text`。

原因是 LLM 预训练通常使用自监督学习，不需要人工提前准备 `label` 字段。训练目标是：

```text
给定前面的 token，预测下一个 token
```

例如文本：

```text
The sky is blue
```

经过 tokenizer 后，可以理解成：

```text
[<|bos|>, The, sky, is, blue]
```

dataloader 会自动构造训练输入和训练目标：

```text
输入 x: [<|bos|>, The, sky, is]
目标 y: [The,     sky, is,  blue]
```

模型看到 `x` 里的当前位置 token，要预测 `y` 里对应位置的下一个 token。所以 `y` 不是数据集中额外提供的人工标签，而是由同一段文本右移一位自动得到的。

## 源码路径

### 1. 从 parquet 读取 `text`

位置：`nanochat/dataset.py`

```python
rg = pf.read_row_group(rg_idx)
texts = rg.column('text').to_pylist()
yield texts
```

`base_data_climbmix` 里的 parquet 文件只有一个核心字段：

```text
text: string
```

每一行是一篇文档字符串。这里还没有 token，也没有 label。

### 2. dataloader 把 `text` 编码成 token

位置：`nanochat/dataloader.py`

```python
token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
```

这一步会把一批字符串编码成 token id，并且在每篇文档前面加 `<|bos|>`。

### 3. dataloader 自动构造输入和目标

位置：`nanochat/dataloader.py`

```python
cpu_inputs.copy_(row_buffer[:, :-1])
cpu_targets.copy_(row_buffer[:, 1:])
```

可以理解成：

```python
# 输入给模型的 token 序列
inputs = tokens[:-1]

# 模型要预测的下一个 token，也就是训练目标
targets = tokens[1:]
```

这就是预训练阶段只需要 `text` 的根本原因：监督信号来自文本自身。

### 4. base_train 执行训练

位置：`scripts/base_train.py`

```python
loss = model(x, y)
loss.backward()
```

这里的 `x` 是输入 token，`y` 是目标 token。

### 5. GPT 内部计算交叉熵 loss

位置：`nanochat/gpt.py`

```python
logits = self.lm_head(x)
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,
    reduction=loss_reduction,
)
```

`logits` 是模型对每个位置的“下一个 token 是词表中每个 token 的概率打分”。`cross_entropy` 会比较模型预测和真实下一个 token，预测越差，loss 越高。

## 问题 2：`<|bos|>` 是什么？有什么作用？

`<|bos|>` 是 Beginning Of Sequence，意思是“序列开始”。

在 nanochat 中，它定义在 `nanochat/tokenizer.py`：

```python
SPECIAL_TOKENS = [
    "<|bos|>",
    ...
]
```

预训练时，每篇文档前面都会加一个 `<|bos|>`。它主要有几个作用：

### 1. 告诉模型“一篇新文档开始了”

训练数据是很多网页、文章、代码片段拼起来的。如果不加边界标记，模型可能把两篇完全无关的文档当成连续上下文。

例如没有 `<|bos|>` 时：

```text
... first document ends. Second document starts ...
```

模型会误以为 `Second document starts` 是上一篇文章的自然延续。

加上 `<|bos|>` 后：

```text
... first document ends. <|bos|> Second document starts ...
```

模型能学到：`<|bos|>` 后面通常是一篇新文档的开头。

### 2. 给序列第一个位置一个可预测目标

如果一篇文档是：

```text
The sky is blue
```

加 `<|bos|>` 后：

```text
[<|bos|>, The, sky, is, blue]
```

训练样本就是：

```text
输入 x: [<|bos|>, The, sky, is]
目标 y: [The,     sky, is,  blue]
```

这样模型可以学习：

```text
在一篇新文档开始时，后面常见的第一个 token 是什么
```

### 3. 减少不同文档之间的上下文污染

nanochat 的 dataloader 会把文档打包到固定长度序列里。`<|bos|>` 相当于一个轻量边界，让模型知道某些 token 前后不一定属于同一篇文档。

这在 `nanochat/dataloader.py` 的注释里也有体现：每一行训练样本都尽量从 BOS 开始，帮助模型看到更清晰的文档上下文。

## 总结

预训练阶段的数据可以非常简单：

```text
text
```

训练时自动变成：

```text
text -> token ids -> inputs/targets -> next-token loss
```

`<|bos|>` 的作用是给文档开始加边界，让模型知道“新序列开始了”，并让第一个 token 的预测也有明确上下文。

这和后面的 SFT 不同。SFT 要训练聊天能力，所以数据需要包含 user/assistant 结构；而 base model 预训练只是在学语言、知识和续写能力，大量纯文本就足够。
