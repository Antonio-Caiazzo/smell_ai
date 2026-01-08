"use client";
import { useState, useCallback, useRef, useEffect } from "react";
import Header from "../../components/HeaderComponent";
import Footer from "../../components/FooterComponent";
import { detectCallGraph } from "../../utils/api";
import { motion, AnimatePresence } from "framer-motion";
import ReactFlow, {
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    MarkerType,
    Node,
    Edge,
    ConnectionLineType,
    useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";
import "reactflow/dist/style.css";
import dagre from "dagre";
import { toast } from "react-toastify";
import JSZip from "jszip";

// Define Types based on Backend Response
interface Smell {
    function_name: string;
    line: number;
    smell_name: string;
    description: string;
    additional_info: string;
}

interface CallSite {
    file_path: string;
    line: number;
    snippet: string;
}

interface GraphNode {
    id: string;
    name: string;
    module: string;
    package: string;
    type: string;
    file_path: string;
    start_line: number;
    end_line: number;
    source_code?: string;
}

interface GraphEdge {
    source: string;
    target: string;
    call_sites: CallSite[];
}

const CodeBlock = ({ code, smells, startLine }: { code: string, smells: Smell[], startLine: number }) => {
    const lines = code.split('\n');

    return (
        <pre className="font-mono text-xs text-gray-800 bg-white p-2 border rounded overflow-x-auto">
            {lines.map((lineContent, index) => {
                const currentLine = startLine + index;
                // Check if this line has any smell
                const smellOnLine = smells.find(s => s.line === currentLine);

                return (
                    <div
                        key={index}
                        className={`${smellOnLine ? 'bg-red-100' : ''} flex`}
                    >
                        <span className="text-gray-400 w-8 flex-shrink-0 select-none text-right mr-2 border-r pr-1">
                            {currentLine}
                        </span>
                        <span className="flex-grow">{lineContent}</span>
                    </div>
                );
            })}
        </pre>
    );
};

const Modal = ({ isOpen, onClose, title, children }: any) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col"
            >
                <div className="p-4 border-b flex justify-between items-center bg-gray-50">
                    <h2 className="text-xl font-bold text-gray-800">{title}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-700 font-bold text-xl">
                        &times;
                    </button>
                </div>
                <div className="p-6 overflow-y-auto flex-grow bg-white">{children}</div>
                <div className="p-4 border-t bg-gray-50 flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                        Close
                    </button>
                </div>
            </motion.div>
        </div>
    );
};

const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    dagreGraph.setGraph({ rankdir: "TB", ranksep: 50, nodesep: 50 });

    nodes.forEach((node) => {
        const label = node.data.label || "";
        const width = Math.max(180, label.length * 9 + 40);
        dagreGraph.setNode(node.id, { width: width, height: 50 });
    });

    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
    });

    dagre.layout(dagreGraph);

    nodes.forEach((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        node.position = {
            x: nodeWithPosition.x - 90,
            y: nodeWithPosition.y - 25,
        };
    });

    return { nodes, edges };
};



// Component to handle auto-fitting view
const AutoFitter = ({ trigger }: { trigger: number }) => {
    const { fitView } = useReactFlow();

    useEffect(() => {
        // Only trigger if generation has happened (trigger > 0)
        if (trigger > 0) {
            // Short delay to ensure nodes are rendered and measured
            const timer = setTimeout(() => {
                fitView({ padding: 0.2, duration: 800 });
            }, 250);
            return () => clearTimeout(timer);
        }
    }, [trigger, fitView]);

    return null;
};

// ReactFlow component wrapper to handle fitView instance access
const GraphArea = ({ nodes, edges, onNodesChange, onEdgesChange, onNodeClick, onEdgeClick, fitViewTrigger }: any) => {
    return (
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            fitView
            attributionPosition="bottom-right"
        >
            <AutoFitter trigger={fitViewTrigger} />
            <MiniMap />
            <Controls />
            <Background color="#aaa" gap={16} />
        </ReactFlow>
    );
};

export default function CallGraphPage() {
    const [code, setCode] = useState("");
    const [loading, setLoading] = useState(false);
    const folderInputRef = useRef<HTMLInputElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Enable folder selection on the hidden input
    useEffect(() => {
        if (folderInputRef.current) {
            folderInputRef.current.webkitdirectory = true;
        }
    }, []);

    // Graph State
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    // Data State
    const [smells, setSmells] = useState<Smell[]>([]);
    const [rawNodes, setRawNodes] = useState<GraphNode[]>([]);
    const [rawEdges, setRawEdges] = useState<GraphEdge[]>([]);

    // ReactFlow Instance for fitView removed in favor of AutoFitter
    const [fitViewTrigger, setFitViewTrigger] = useState(0);

    // Selection Selection
    const [selectedNodeData, setSelectedNodeData] = useState<any>(null);
    const [selectedEdgeData, setSelectedEdgeData] = useState<any>(null);

    const [fileName, setFileName] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleFolderSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        // Validation if user selects a single file that looks like a zip or py script in "Folder" mode, warn them
        if (files.length === 1) {
            const f = files[0];
            if (f.name.endsWith('.zip') || f.name.endsWith('.py')) {
                toast.error("Please ensure you are uploading a folder, not a single file. For single Python files, use the specific button.");
                e.target.value = ""; // Clear input
                setSelectedFile(null); // Clear any previous file selection
                setFileName(""); // Clear displayed filename
                setLoading(false);
                return;
            }
        }

        setLoading(true);
        try {
            const zip = new JSZip();
            let hasFiles = false;
            let folderName = "project";

            // files is a Flat list, but has webkitRelativePath
            Array.from(files).forEach((file: any) => {
                // Filter for valid files (e.g. .py)
                if (file.name.endsWith('.py')) {
                    hasFiles = true;
                    zip.file(file.webkitRelativePath, file);
                    if (folderName === "project") {
                        folderName = file.webkitRelativePath.split('/')[0];
                    }
                }
            });

            if (!hasFiles) {
                toast.error("No Python files found in the selected folder.");
                setLoading(false);
                return;
            }

            const content = await zip.generateAsync({ type: "blob" });
            const zipFile = new File([content], `${folderName}.zip`, { type: "application/zip" });

            setSelectedFile(zipFile);
            setFileName(`${folderName} (Folder)`);
            setCode(""); // Clear preview
            toast.info(`Folder packed as ${folderName}.zip`);

        } catch (error) {
            console.error(error);
            toast.error("Failed to process folder.");
        } finally {
            setLoading(false);
            if (folderInputRef.current) {
                folderInputRef.current.value = "";
            }
            // Also clear file input to ensure clean state for next single file upload
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        }
    };

    // Single File Selection
    const handleSingleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setFileName(file.name);

            // Clear Code Editor
            if (file.name.endsWith(".py")) {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    setCode(ev.target?.result as string);
                };
                reader.readAsText(file);
            } else {
                setCode("");
            }
            toast.info(`Loaded: ${file.name}`);

            // Allow re-selecting the same file if needed by clearing the input value after processing
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile && !code.trim()) {
            toast.error("Please upload a file or enter python code.");
            return;
        }
        setLoading(true);
        setNodes([]);
        setEdges([]);

        try {
            const payload = selectedFile ? selectedFile : code;
            const result = await detectCallGraph(payload);

            if (result.success) {
                setSmells(result.smells || []);
                processGraph(result.call_graph, result.smells || []);
                toast.success("Analysis Complete!");
            } else {
                const msg = result.message || "Unknown error occurred";
                toast.error("Analysis Failed: " + msg);
            }
        } catch (e: any) {
            toast.error("An error occurred: " + (e.message || e));
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const processGraph = (graphData: any, smellList: Smell[]) => {
        if (!graphData) return;

        const gNodes: GraphNode[] = graphData.nodes || [];
        const gEdges: GraphEdge[] = graphData.edges || [];
        setRawNodes(gNodes);
        setRawEdges(gEdges);

        const rfNodes: Node[] = gNodes.map((node) => {
            const nodeSmells = smellList.filter(
                (s) => {
                    const nodePathNormalized = node.file_path ? node.file_path.normalize('NFC') : "";
                    const smellRaw = s as any;

                    if (smellRaw.file) {
                        const smellFileNormalized = smellRaw.file.normalize('NFC');

                        const match = (smellFileNormalized === nodePathNormalized) ||
                            (nodePathNormalized.endsWith(smellFileNormalized)) ||
                            (smellFileNormalized.endsWith(nodePathNormalized));

                        if (!match) return false;
                    } else if (smellRaw.filename) {
                        const smellFileNormalized = smellRaw.filename.normalize('NFC');

                        if (!nodePathNormalized.endsWith(smellFileNormalized) && !smellFileNormalized.endsWith(nodePathNormalized)) {
                            return false;
                        }
                    } else {

                        return false;
                    }

                    return s.function_name === node.name || (s.line >= node.start_line && s.line <= node.end_line);
                }
            );
            const hasSmell = nodeSmells.length > 0;

            return {
                id: node.id,
                data: {
                    label: `${node.file_path ? node.file_path.split('/').pop() + ": " : ""}${node.name}`,
                    raw: node,
                    smells: nodeSmells
                },
                position: { x: 0, y: 0 },
                style: {
                    background: hasSmell ? "#fee2e2" : "#f0f9ff",
                    border: hasSmell ? "2px solid #ef4444" : "1px solid #7dd3fc",
                    color: "#333",
                    width: 'auto',
                    minWidth: 180,
                },
            };
        });

        const rfEdges: Edge[] = gEdges.map((edge) => ({
            id: `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            type: "smoothstep",
            markerEnd: { type: MarkerType.ArrowClosed },
            data: { call_sites: edge.call_sites },
            animated: true,
        }));

        const layouted = getLayoutedElements(rfNodes, rfEdges);
        setNodes(layouted.nodes);
        setEdges(layouted.edges);
        setFitViewTrigger((prev) => prev + 1);
    };

    const onNodeClick = (_: any, node: Node) => {
        setSelectedNodeData(node.data);
    };

    const onEdgeClick = (_: any, edge: Edge) => {
        setSelectedEdgeData(edge.data);
    };

    return (
        <div className="min-h-screen flex flex-col bg-gray-50">
            <Header />
            <main className="flex-grow flex flex-col h-[calc(100vh-64px)] pt-24">
                <div className="flex flex-row h-full">

                    {/* Left Panel: Input */}
                    <div className="w-1/3 p-6 border-r bg-white flex flex-col shadow-lg z-10">
                        <h2 className="text-2xl font-bold mb-4 text-blue-700">Upload File / Project</h2>

                        <div className="flex-grow flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors relative">
                            <input
                                ref={fileInputRef}
                                data-testid="file-upload-input"
                                type="file"
                                accept=".py,.zip"
                                onChange={handleSingleFileSelect}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            <div className="text-center p-4">
                                <p className="text-gray-500 font-medium text-lg">Click or Drag to Upload</p>
                                <p className="text-gray-400 text-sm mt-2">.py file or .zip project</p>
                            </div>
                        </div>

                        {/* Folder Upload Option */}
                        <div className="text-center mb-4">
                            <span className="text-gray-400 text-sm">or</span>
                            <button
                                onClick={() => folderInputRef.current?.click()}
                                className="ml-2 text-blue-600 hover:text-blue-800 text-sm font-medium hover:underline focus:outline-none"
                            >
                                Upload a Folder
                            </button>
                            <input
                                ref={folderInputRef}
                                type="file"
                                onChange={handleFolderSelect}
                                className="hidden"
                                multiple
                            />
                        </div>

                        {fileName && (
                            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded text-sm text-green-800 text-center">
                                <strong>Loaded File:</strong> {fileName}
                            </div>
                        )}

                        <button
                            onClick={handleAnalyze}
                            disabled={loading || (!code && !selectedFile)}
                            className={`mt-2 py-3 rounded-lg text-white font-bold text-lg shadow-md transition-all transform hover:scale-105 ${loading || (!code && !selectedFile) ? "bg-gray-400 cursor-not-allowed" : "bg-purple-600 hover:bg-purple-700"
                                }`}
                        >
                            {loading ? "Analyzing..." : "Generate Call Graph"}
                        </button>
                    </div>

                    {/* Right Panel: Graph */}
                    <div className="w-2/3 bg-gray-100 relative">
                        <GraphArea
                            nodes={nodes}
                            edges={edges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            onNodeClick={onNodeClick}
                            onEdgeClick={onEdgeClick}
                            fitView
                            fitViewTrigger={fitViewTrigger}
                            attributionPosition="bottom-right"
                        />
                        <div className="absolute top-4 right-4 bg-white p-4 rounded-lg shadow-md border border-gray-200 opacity-95">
                            <h4 className="font-bold text-gray-700 mb-2 border-b pb-1">Legend</h4>
                            <div className="flex items-center mb-1">
                                <div className="w-4 h-4 rounded bg-[#fee2e2] mr-2 border border-[#ef4444]"></div>
                                <span className="text-sm text-gray-600">Smell Detected</span>
                            </div>
                            <div className="flex items-center mb-3">
                                <div className="w-4 h-4 rounded bg-[#f0f9ff] mr-2 border border-[#7dd3fc]"></div>
                                <span className="text-sm text-gray-600">No smells</span>
                            </div>
                            <div className="text-sm font-semibold text-gray-700">
                                Total Smells: <span className="text-red-600">{smells.length}</span>
                            </div>
                            <div className="mt-2 text-xs text-gray-500 pt-2 border-t">
                                <p>Click node for details</p>
                                <p>Click edge for calls</p>
                            </div>
                        </div>
                    </div>

                </div>
            </main>

            {/* Node Details Modal */}
            <AnimatePresence>
                {selectedNodeData && (
                    <Modal
                        isOpen={!!selectedNodeData}
                        onClose={() => setSelectedNodeData(null)}
                        title={`Node: ${selectedNodeData.raw.name}`}
                    >
                        <div className="space-y-4">
                            <div>
                                <h3 className="font-semibold text-gray-700">Type:</h3>
                                <p>{selectedNodeData.raw.type}</p>
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-700">Module/Package:</h3>
                                <p className="break-all">{selectedNodeData.raw.package || "root"} / {selectedNodeData.raw.module}</p>
                            </div>
                            <div>
                                <h3 className="font-semibold text-gray-700">Defined in:</h3>
                                <p className="font-mono text-sm text-gray-600 bg-gray-50 p-1 rounded inline-block break-all">
                                    {selectedNodeData.raw.file_path} : {selectedNodeData.raw.start_line}-{selectedNodeData.raw.end_line}
                                </p>
                            </div>
                            <div className="bg-gray-100 p-3 rounded font-mono text-sm border-l-4 border-blue-400">
                                <p className="mb-2 font-bold text-gray-600">Source Code (Lines {selectedNodeData.raw.start_line} - {selectedNodeData.raw.end_line}):</p>
                                <div className="">
                                    {selectedNodeData.raw.source_code ? (
                                        <CodeBlock
                                            code={selectedNodeData.raw.source_code}
                                            smells={selectedNodeData.smells}
                                            startLine={selectedNodeData.raw.start_line}
                                        />
                                    ) : (
                                        <span className="italic text-gray-400">Source code not available</span>
                                    )}
                                </div>
                            </div>

                            <div>
                                <h3 className="font-semibold text-red-600 flex items-center">
                                    Detected Smells ({selectedNodeData.smells.length})
                                </h3>
                                {selectedNodeData.smells.length === 0 ? (
                                    <p className="text-gray-500 italic">No smells detected on this node.</p>
                                ) : (
                                    <ul className="space-y-3 mt-2">
                                        {[...selectedNodeData.smells].sort((a: Smell, b: Smell) => a.line - b.line).map((s: Smell, idx: number) => (
                                            <li key={idx} className="bg-red-50 p-3 rounded border border-red-200">
                                                <p className="font-bold text-red-700">{s.smell_name}</p>
                                                <p className="text-sm text-gray-700 mt-1">{s.description}</p>
                                                <p className="text-xs text-gray-500 mt-1">Line: {s.line}</p>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        </div>
                    </Modal>
                )}
            </AnimatePresence>

            {/* Edge Details Modal */}
            <AnimatePresence>
                {selectedEdgeData && (
                    <Modal
                        isOpen={!!selectedEdgeData}
                        onClose={() => setSelectedEdgeData(null)}
                        title="Call Details"
                    >
                        <div>
                            <h3 className="font-semibold text-gray-700 mb-2">Call Sites:</h3>
                            {selectedEdgeData.call_sites && selectedEdgeData.call_sites.length > 0 ? (
                                <ul className="space-y-2">
                                    {selectedEdgeData.call_sites.map((site: CallSite, idx: number) => (
                                        <li key={idx} className="bg-gray-100 p-3 rounded border border-gray-300 font-mono text-sm">
                                            <div className="flex justify-between text-xs text-gray-500 mb-1">
                                                <span>{site.file_path}</span>
                                                <span>Line {site.line}</span>
                                            </div>
                                            <div className="text-blue-800 font-semibold">
                                                {site.snippet || "Code snippet not available"}
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-gray-500 italic">No specific call sites recorded.</p>
                            )}
                        </div>
                    </Modal>
                )}
            </AnimatePresence>
            <Footer />
        </div>
    );
}
