import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    let c0 = commentary(wt, null, 0)`

最后，我们来到模型的末端。最后变换器模块的输出经过层归一化处理，然后我们使用线性变换（矩阵乘法），这次没有偏置。

这种最终转换将我们的每个列向量从长度 C 转换为长度 nvocab。因此，它实际上为我们的每一列中的每个词汇生成一个分数。
这些分数有一个特殊的名称：logits。

"logits"这个名字来源于 "log-odds"，即每个标记（token）的几率的对数。之所以使用 "对数"，
是因为我们接下来应用的 softmax 会进行指数运算，将其转换为 "几率 "或概率。

为了将这些分数转换为良好的概率，我们对它们进行 softmax 操作。现在，对于每一列，
我们都有一个模型分配给词汇中每个单词的概率。

在这个特定的模型中，它已经有效地学习了如何对三个字母进行排序这一问题的所有答案，
因此概率在很大程度上倾向于正确答案。

当我们对模型进行时间步进时，我们会使用上一列的概率来决定下一个要添加到序列中的标记（token）。
例如，如果我们已经向模型提供了 6 个标记，我们就会使用第 6 列的输出概率。

这一列的输出是一系列概率，我们实际上必须从中选出一个作为下一个概率。我们的做法是 "从分布中采样"。
也就是说，我们按照概率加权随机选择一个标记。例如，概率为 0.9 的令牌将在 90% 的情况下被选中。

不过，这里还有其他选择，比如总是选择概率最高的标记。

我们还可以使用温度参数来控制分布的 "平滑度"。温度越高，分布越均匀，温度越低，分布越集中在概率最高的标记上。

在应用 softmax 之前，我们先用温度除以 logits（线性变换的输出）。由于 softmax 中的指数化会对较大的数字产生较大影响，因此将所有数字拉近会减少这种影响。
`;

}
