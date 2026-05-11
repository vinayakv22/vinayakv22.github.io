import { useEffect, useRef, useState } from "react";

const MANAGED_ATTR = "data-canva-managed";

function cloneNodeWithAttributes(node) {
  const element = document.createElement(node.tagName.toLowerCase());

  for (const { name, value } of [...node.attributes]) {
    if (name === "nonce") {
      continue;
    }

    element.setAttribute(name, value);
  }

  if (node.textContent) {
    element.textContent = node.textContent;
  }

  element.setAttribute(MANAGED_ATTR, "true");
  return element;
}

function removeManagedNodes() {
  document.querySelectorAll(`[${MANAGED_ATTR}="true"]`).forEach((node) => {
    node.remove();
  });
}

export default function App() {
  const mountRef = useRef(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadCanvaSite() {
      try {
        const response = await fetch("/canva-export.html", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Failed to load Canva export (${response.status})`);
        }

        const html = await response.text();
        if (cancelled) {
          return;
        }

        const parsed = new DOMParser().parseFromString(html, "text/html");
        const mountNode = mountRef.current;

        if (!mountNode) {
          return;
        }

        removeManagedNodes();
        mountNode.innerHTML = "";

        document.documentElement.lang = parsed.documentElement.lang || "en";
        document.documentElement.dir = parsed.documentElement.dir || "ltr";
        document.documentElement.className = parsed.documentElement.className;

        const headNodes = [...parsed.head.children].filter((node) => {
          if (node.tagName.toLowerCase() === "base") {
            return false;
          }

          if (
            node.tagName.toLowerCase() === "script" &&
            node.textContent.includes("window.location.protocol===")
          ) {
            return false;
          }

          return true;
        });

        headNodes.forEach((node) => {
          document.head.appendChild(cloneNodeWithAttributes(node));
        });

        const bodyNodes = [...parsed.body.childNodes];
        bodyNodes.forEach((node) => {
          if (node.nodeType === Node.TEXT_NODE && !node.textContent.trim()) {
            return;
          }

          if (node.nodeType === Node.ELEMENT_NODE && node.tagName.toLowerCase() === "script") {
            mountNode.appendChild(cloneNodeWithAttributes(node));
            return;
          }

          mountNode.appendChild(node.cloneNode(true));
        });

        setStatus("ready");
      } catch (loadError) {
        if (!cancelled) {
          setStatus("error");
          setError(loadError instanceof Error ? loadError.message : "Unknown error");
        }
      }
    }

    loadCanvaSite();

    return () => {
      cancelled = true;
      removeManagedNodes();
    };
  }, []);

  return (
    <>
      {status !== "ready" && (
        <div className="app-shell">
          <div className="app-card">
            <p className="eyebrow">React Refactor</p>
            <h1>{status === "error" ? "Could not load the Canva export." : "Loading portfolio..."}</h1>
            <p>{status === "error" ? error : "Rebuilding the original site inside the React app."}</p>
          </div>
        </div>
      )}
      <div ref={mountRef} />
    </>
  );
}
