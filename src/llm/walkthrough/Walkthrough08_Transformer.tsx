import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    let c0 = commentary(wt, null, 0)`
    这就是一个完整的变换器(transformer)模块！

    它们构成了任何 GPT 模型的主体，并且会重复多次，一个区块的输出会输入到下一个区块，继续剩余路径。

    在深度学习中，很难确切地说出这些层中的每一层都在做什么，但我们有一些大致的想法：较早的层往往
    专注于学习较低级别的特征和模式，而较后的层则学习识别和理解更高级别的抽象和关系。
    在自然语言处理中，低层可能学习语法、句法和简单的词汇关联，而高层可能捕捉更复杂的语义关系、
    话语结构和上下文相关的含义。
    `;

}
