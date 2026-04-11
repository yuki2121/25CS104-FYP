import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import PoseEditorCanvas from "./PoseEditorCanvas.tsx";
import ResultDisplay from "./ResultDisplay.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div>
      <aside
        style={{
          position: "fixed", 
          top: 0,
          left: 0,
          width: "520px", 
          height: "100vh", 
          overflowY: "auto", 
          borderRight: "1px solid #ccc",
          padding: "10px",
          backgroundColor: "#fff", 
          zIndex: 10, 
          textAlign: "center",
        }}
      >
        <PoseEditorCanvas />
      </aside>

      <main
        id = "result-scroll-root"
        style={{
          marginLeft: "550px", 
          height: "100vh", 
          overflowY: "auto",
          boxSizing: "border-box",
        }}
      >
        <ResultDisplay />
      </main>
    </div>
  </StrictMode>,
);
