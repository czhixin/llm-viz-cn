import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough06_Projection(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.Input_Detail_Projection) {
        return;
    }

    setInitialCamera(state, new Vec3(-73.167, 0.000, -270.725), new Vec3(293.606, 2.613, 1.366));
    let block = layout.blocks[0];
    wt.dimHighlightBlocks = [...block.heads.map(h => h.vOutBlock), block.projBias, block.projWeight, block.attnOut];

    let outBlocks = block.heads.map(h => h.vOutBlock);

    commentary(wt, null, 0)`

在自关注力（self-attention）过程之后，我们会从每个头部得到输出。这些输出是受 Q 和 K 向量影响而适度混合的 V 向量。

要合并每个头部的${c_blockRef('输出向量(output vectors)', outBlocks)}，我们只需将它们堆叠在一起即可。因此，在时间 ${c_dimRef('t = 4', DimStyle.T)} 时，我们将从 3 个长度为 ${c_dimRef('A = 16', DimStyle.A)} 的向量叠加到 1 个长度为 ${c_dimRef('C = 48', DimStyle.C)} 的向量。
`;

    breakAfter();

    let t_fadeOut = afterTime(null, 1.0, 0.5);
    // let t_zoomToStack = afterTime(null, 1.0);
    let t_stack = afterTime(null, 1.0);

    breakAfter();

    commentary(wt)`

值得注意的是，在 GPT 中，头部 (${c_dimRef('A = 16', DimStyle.A)}) 内向量的长度等于 ${c_dimRef('C', DimStyle.C)}  / num_heads。这确保了当我们将它们重新堆叠在一起时，能得到原来的长度 ${c_dimRef('C', DimStyle.C)}。

在此基础上，我们进行投影，得到该层的输出。这是一个简单的矩阵-向量乘法，以每列为单位，并加上偏置。
`;

    breakAfter();

    let t_process = afterTime(null, 3.0);

    breakAfter();

    commentary(wt)`

现在我们得到了自注意力(self-attention)层的输出。我们不是直接将这个输出传递到下一个阶段，而是将其与输入嵌入(input embedding)进行元素级相加。这个过程，用绿色垂直箭头表示，被称为 _残差连接（residual connection）_ 或 _残差路径（residual pathway）_。
`;

    breakAfter();

    let t_zoomOut = afterTime(null, 1.0, 0.5);
    let t_processResid = afterTime(null, 3.0);

    cleanup(t_zoomOut, [t_fadeOut, t_stack]);

    breakAfter();

    commentary(wt)`
    就像层归一化一样，残差路径对于在深度神经网络中实现有效学习非常重要。

    现在有了自注意力的结果，我们可以将其传递到变换器（transformer）的下一部分：前馈网络(the feed-forward network)。
`;

    breakAfter();

    if (t_fadeOut.active) {
        for (let head of block.heads) {
            for (let blk of head.cubes) {
                if (blk !== head.vOutBlock) {
                    blk.opacity = lerpSmoothstep(1, 0, t_fadeOut.t);
                }
            }
        }
    }

    if (t_stack.active) {
        let targetZ = block.attnOut.z;
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            let head = block.heads[headIdx];
            let targetY = head.vOutBlock.y + head.vOutBlock.dy * (headIdx - block.heads.length + 1);
            head.vOutBlock.y = lerp(head.vOutBlock.y, targetY, t_stack.t);
            head.vOutBlock.z = lerp(head.vOutBlock.z, targetZ, t_stack.t);
        }
    }

    let processInfo = startProcessBefore(state, block.attnOut);

    if (t_process.active) {
        processUpTo(state, t_process, block.attnOut, processInfo);
    }

    moveCameraTo(state, t_zoomOut, new Vec3(-8.304, 0.000, -175.482), new Vec3(293.606, 2.623, 2.618));

    if (t_processResid.active) {
        processUpTo(state, t_processResid, block.attnResidual, processInfo);
    }
}
