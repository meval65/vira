import { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { api } from '../api/client';
import type { Triple } from '../api/types';
import { useWebSocket } from '../hooks/useWebSocket';

export default function NeuralViz() {
    const fgRef = useRef<any>(null);
    const [graphData, setGraphData] = useState<{ nodes: any[], links: any[] }>({ nodes: [], links: [] });
    const { lastMessage } = useWebSocket('/ws');

    useEffect(() => {
        loadGraph();
    }, []);

    useEffect(() => {
        if (lastMessage && (lastMessage as any).type === 'triple_update') {
            loadGraph();
        }
    }, [lastMessage]);

    const loadGraph = async () => {
        try {
            const triples = await api.getTriples(500);

            const nodes: any[] = [];
            const links: any[] = [];
            const nodeIds = new Set();

            triples.forEach((t: Triple) => {
                if (!nodeIds.has(t.subject)) {
                    nodes.push({ id: t.subject, group: 'subject', val: 5 });
                    nodeIds.add(t.subject);
                }
                if (!nodeIds.has(t.object)) {
                    nodes.push({ id: t.object, group: 'object', val: 3 });
                    nodeIds.add(t.object);
                }
                links.push({
                    source: t.subject,
                    target: t.object,
                    name: t.predicate
                });
            });

            setGraphData({ nodes, links });
        } catch (error) {
            console.error("Failed to load graph data", error);
        }
    };

    return (
        <div className="w-full h-full rounded-2xl overflow-hidden glass-panel relative">
            <div className="absolute top-4 left-4 z-10 bg-black/40 px-3 py-1 rounded-full border border-white/10 pointer-events-none">
                <span className="text-xs font-mono text-cyan-400 font-bold animate-pulse">NEURAL_VIZ_ACTIVE</span>
            </div>
            <ForceGraph2D
                ref={fgRef}
                graphData={graphData}
                nodeLabel="id"
                nodeColor={node => (node as any).group === 'subject' ? '#8b5cf6' : '#ec4899'}
                linkColor={() => '#ffffff30'}
                backgroundColor="transparent"
                width={800}
                height={500}
                onNodeClick={node => {
                    fgRef.current?.centerAt((node as any).x, (node as any).y, 1000);
                    fgRef.current?.zoom(3, 1000);
                }}
            />
        </div>
    );
}
