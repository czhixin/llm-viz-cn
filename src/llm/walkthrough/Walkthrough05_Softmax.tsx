import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough05_Softmax(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Softmax) {
        return;
    }

    setInitialCamera(state, new Vec3(-24.350, 0.000, -1702.195), new Vec3(283.100, 0.600, 1.556));

    let c0 = commentary(wt, null, 0)`

Softmax操作不仅在前面的部分中作为自注意力的一部分使用，而且也会出现在模型的最后。

它的目的是将一个向量的值归一化，使其总和为 1.0。然而，这并不像除以总和那么简单。相反，每个输入值都要先进行指数化处理。

  a = exp(x_1)

这样做的效果是使所有值都为正。有了指数化值的向量后，我们就可以用每个值除以所有值的总和。
这将确保所有数值之和为 1.0。由于所有指数化值都是正值，我们知道得出的值将介于 0.0 和 1.0 之间，
这就为原始值提供了一个概率分布。

这就是 softmax 的原理：简单地将数值指数化，然后除以总和。

不过，还有一个小问题。如果输入值很大，那么指数化后的值也会很大。我们最终会用一个很大的数除以一个很大的数，
这可能会导致浮点运算出现问题。

Softmax 运算的一个有用特性是，如果我们在所有输入值上添加一个常数，结果将是相同的。
因此，我们可以找到输入向量中的最大值，然后将其从所有值中减去。这样就能确保最大值为 0.0，
从而使 softmax 在数值上保持稳定。

让我们来看看自注意力层中的 softmax 操作。我们的输入向量是自注意力矩阵的一行（但只到对角线）。

与层归一化一样，我们有一个中间步骤来存储一些聚合值以保持流程效率。

对于每一行，我们都会存储该行的最大值以及移位值和指数值的总和。
然后，为了生成相应的输出行，我们可以执行一小套操作：减去最大值、指数化和除以总和。

为什么叫 "softmax"？这种操作的 "硬 "版本称为 argmax，简单地说，就是找到最大值，
将其设为 1.0，并将所有其他值设为 0.0。相比之下，softmax操作则是该操作的 "更柔和 "版本。
由于softmax涉及指数运算，最大值被强调并推向 1.0。同时仍保持所有输入值的概率分布。
这样就能获得更细致的表示，不仅能捕捉到最有可能的选项，还能捕捉到其他选型的相对可能性。
`;

}
