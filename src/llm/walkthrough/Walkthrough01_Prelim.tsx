import React from 'react';
import { Phase } from "./Walkthrough";
import { commentary, embed, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Vec3 } from '@/src/utils/vector';

let minGptLink = 'https://github.com/karpathy/minGPT';
let pytorchLink = 'https://pytorch.org/';
let andrejLink = 'https://karpathy.ai/';
let zeroToHeroLink = 'https://karpathy.ai/zero-to-hero.html';

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    setInitialCamera(state, new Vec3(184.744, 0.000, -636.820), new Vec3(296.000, 16.000, 13.500));

    let c0 = commentary(wt, null, 0)`
在深入了解算法的复杂性之前，我们先来做个简单的回顾。

本指南侧重于 _推理_ 而非训练，因此只是整个机器学习过程的一小部分。在我们的例子中，模型的权重已经预先训练好，我们使用推理过程来生成输出。这可以直接在浏览器中运行。
在我们的例子中，模型的权重已经预先训练好，我们使用推理过程来生成输出。这可以直接在浏览器中运行。

这里展示的模型是 GPT（生成式预训练转换器）系列的一部分，可以说是 "基于上下文的标记预测器"。OpenAI 在 2018 年引入了这一家族，其著名成员包括 GPT-2、GPT-3 和 GPT-3.5 Turbo，后者是广泛使用的 ChatGPT 的基础。它还可能与 GPT-4 有关，但具体细节仍不得而知。

本指南受到 ${embedLink('minGPT', minGptLink)} GitHub 项目的启发，该项目是 ${embedLink('Andrej Karpathy', andrejLink)} 在 ${embedLink('PyTorch', pytorchLink)} 中创建的最小 GPT 实现。他的 YouTube ${embedLink("Neural Networks: Zero to Hero", zeroToHeroLink)} 系列和 minGPT 项目是创建本指南的宝贵参考资源。这里介绍的玩具模型就是基于 minGPT 项目中的一个模型。

好的，让我们开始吧！
`;

}

export function embedLink(a: React.ReactNode, href: string) {
    return embedInline(<a className={s.externalLink} href={href} target="_blank" rel="noopener noreferrer">{a}</a>);
}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}


// Another similar model is BERT (bidirectional encoder representations from transformers), a "context-aware text encoder" commonly
// used for tasks like document classification and search.  Newer models like Facebook's LLaMA (large language model architecture), continue to use
// a similar transformer architecture, albeit with some minor differences.
