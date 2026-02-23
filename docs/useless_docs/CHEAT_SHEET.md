# Cheat Sheet: 每个 Ticket 你到底做什么

把这个贴在显示器旁边。

---

## 你每个 Ticket 只做 4 件事

| # | 你做什么 | 花多久 |
|---|---------|--------|
| 1 | 用自己的话复述概念（60秒） | 1 min |
| 2 | 解释代码为什么这样写（3 点：dataflow、shape、关键行） | 2 min |
| 3 | 遇到意外时判断：阻塞了吗？（见下方判断规则） | 10 sec |
| 4 | 在 PROJECT_LOG 的「我的理解」里写 2-5 句 | 3 min |

其余全部让 AI 做。

---

## 流程顺序

```
你发提示词 ①
    ↓
Claude 讲解概念（不写代码）  这里修改为先生成代码
Claude 问你 2 个问题        我问gpt问题
    ↓
✅ 你回答（复述概念，60秒）   让gpt问我问题
    ↓
Claude 纠正你 → 如果没问题：  
    ↓
你发提示词 ② 让claude 写代码 / 他可能已经写过了 看一下集成的是否可用
    ↓
Claude 写代码
    ↓
  ┌─── 实现过程中发现意外？──────────────────────┐
  │                                               │
  │  问自己："不修这个，当前 ticket 能跑通吗？"      │
  │                                               │
  │  能跑通 → 不动手，记到 Discoveries（B 或 C 类） │
  │           B类：未来需要 → 写进 TICKETS 的 Backlog │
  │           C类：旧代码有 bug → 建补丁 ticket      │
  │                                               │
  │  跑不通 → 当场最小修复，记到 Discoveries（A 类）  │
  │           只修到能跑通为止，不展开               │
  │                                               │
  └───────────────────────────────────────────────┘
    ↓
Claude 让你解释代码
    ↓
✅ 你解释（3 个要点：dataflow / shape / 关键行）
    ↓
Claude 纠正你 → 如果没问题：
    ↓
你发提示词 ③
    ↓
Claude 用中文更新 PROJECT_LOG（含发现）
    ↓
✅ 你去 PROJECT_LOG 补上「我的理解」（2-5 句）
    ↓
Done. 开始下一个 Ticket.

---
每 3 个 Ticket 多加一步：
    ↓
你发提示词 ④
    ↓
Claude 考你 → 你答 → Claude 指出盲区 → 你重新解释 → 写入「盲区」
```

---

## 提示词（只有 4 句，复制粘贴就行）

### ① 开场
```
Read docs/AI_COLLAB_PROTOCOL.md, TICKETS.md, PROJECT_LOG.md.
Starting T0X. Explain the mental model first (dataflow + tensor shapes).
Do NOT write code yet. Ask me  suitable questions after explaining.
```

### ② 推进实现
```
Proceed to implement T0X. Keep it minimal, single file, explicit,
comments on shape changes. After code, ask me to explain the
implementation back to you.
If we hit unexpected issues, follow the Discovery protocol in
AI_COLLAB_PROTOCOL rule 8.
```

### ③ 记录
```
Update docs/PROJECT_LOG.md for T0X using the standard format.
Write in Chinese. Include any Discoveries (A/B/C).
If there are B-type items, add them to the Backlog in TICKETS.md.
If there are C-type items, create a patch ticket in TICKETS.md.
Leave "我的理解" for me to fill.
```

### ④ 每 3 个 Ticket 的 Quiz
```
Quiz me on T01-T03. Ask hard conceptual + implementation questions.
Identify blind spots. Reteach weak parts. Update 盲区 in
PROJECT_LOG.md in Chinese. Re-ask until I can explain clearly.
```

---

## Discovery 判断规则（一条就够）

> **"不修这个，当前 ticket 能跑通吗？"**
>
> - 能 → 记下来，不动手
> - 不能 → 最小修复，记 A 类

---

## 语言规则

| 文件 | 语言 | 原因 |
|---|---|---|
| AI_COLLAB_PROTOCOL | 英文 | 指令，要求 Claude 精确遵守 |
| TICKETS | 英文 | 指令，要求 Claude 精确读取 |
| PROJECT_LOG | 中文 | 读者是你 |
| 对话 | 中文 | 你的母语 |

---

## 关键原则

- 如果你跳过复述，这个系统就退化成"AI 代写"
-「我的理解」不能空着，哪怕只写"我还是不太懂 X"也算
-「发现」不能空着，哪怕写"无"也要写，强迫自己想一下
- PROJECT_LOG 超过 5 个 ticket → 旧的搬到 ARCHIVE，只保留最近 5 个
- Ticket 控制方向，Discoveries 处理现实。计划 ≠ 僵化。
