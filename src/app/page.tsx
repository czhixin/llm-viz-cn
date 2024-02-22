import React from 'react';
import { LayerView } from '@/src/llm/LayerView';
import { InfoButton } from '@/src/llm/WelcomePopup';

export const metadata = {
  title: 'LLM 可视化',
  description: 'LLM的3D动画可视化演练',
};

import { Header } from '@/src/homepage/Header';

export default function Page() {
    return <>
        <Header title="大语言模型(LLM)可视化">
            <InfoButton />
        </Header>
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
