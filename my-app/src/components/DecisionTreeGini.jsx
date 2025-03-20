import React, { useState, useEffect } from "react";
import "./DecisionTreeGini.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";


const DecisionTreeGini = ({ fileContent }) => {
  const [features, setFeatures] = useState("");
  const [target, setTarget] = useState("");
  const [treeOutput, setTreeOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState("gini");  // Lưu phương pháp (gini hoặc entropy)
  const [steps, setSteps] = useState([]);

  useEffect(() => {
    if (treeOutput) {
      const scripts = document.querySelectorAll(".results script");
      scripts.forEach((script) => {
        const newScript = document.createElement("script");
        newScript.textContent = script.textContent;
        script.parentNode.replaceChild(newScript, script);
      });
    }
  }, [treeOutput]);

  const handleDecisionTreeClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }

    if (!features || !target) {
      alert("Vui lòng nhập các cột đặc trưng (features) và cột mục tiêu (target).");
      return;
    }

    setLoading(true);
    setTreeOutput(null);
    setSteps([]);

    // Chuẩn bị dữ liệu cho API
    const sanitizedFileContent = fileContent.map((item) => {
      if (typeof item !== "object" || item === null) {
        console.error("Dữ liệu không hợp lệ:", item);
        throw new Error("Dữ liệu trong file không đúng định dạng.");
      }

      const sanitizedItem = { ...item };
      for (const key in sanitizedItem) {
        if (
          sanitizedItem[key] === Infinity ||
          sanitizedItem[key] === -Infinity ||
          Number.isNaN(sanitizedItem[key])
        ) {
          sanitizedItem[key] = null;
        }
      }
      return sanitizedItem;
    });

    const requestBody = {
      data: sanitizedFileContent, 
      target_column: target.trim(),
      feature_columns: features.split(",").map((f) => f.trim()),
      method: method, // Gửi phương pháp (gini hoặc entropy)
    };

    // Gửi yêu cầu API
    fetch("http://127.0.0.1:8000/myapp/decision_tree_view/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data && data.plotly_html) {
          setTreeOutput(data.plotly_html);
          setSteps(data.steps || []);
        } else if (data.error) {
          console.error("Lỗi từ API:", data.error);
          alert(`Lỗi từ API: ${data.error}`);
        } else {
          console.error("Kết quả từ API không hợp lệ");
        }
      })
      .catch((error) => {
        console.error("Lỗi khi gửi yêu cầu Decision Tree:", error);
      })
      .finally(() => setLoading(false)); // Tắt trạng thái loading
  };

  return (
    <div className="decision-tree-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Cây Quyết Định (Gini hoặc Entropy)</h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        Chọn phương pháp chia tách dữ liệu: Gini hoặc Entropy. Gini giúp phân tích sự bất đồng trong nhóm, trong khi Entropy đo độ hỗn loạn trong dữ liệu.
      </p>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="features" style={{ marginRight: "10px" }}>
          Nhập <strong>các cột đặc trưng (features)</strong> (phân cách bằng dấu phẩy):
        </label>
        <input
          type="text"
          id="features"
          value={features}
          onChange={(e) => setFeatures(e.target.value)}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="target" style={{ marginRight: "10px" }}>
          Nhập <strong>cột mục tiêu (target)</strong>:
        </label>
        <input
          type="text"
          id="target"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={() => setMethod("gini")}
          style={{
            padding: "10px 20px",
            backgroundColor: method === "gini" ? "#4CAF50" : "#ccc",
            color: "#fff",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          Gini
        </button>
        <button
          onClick={() => setMethod("entropy")}
          style={{
            padding: "10px 20px",
            backgroundColor: method === "entropy" ? "#4CAF50" : "#ccc",
            color: "#fff",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
            marginLeft: "10px",
          }}
        >
          Entropy
        </button>
      </div>

      <button
        onClick={handleDecisionTreeClick}
        disabled={loading}
        style={{
          padding: "10px 20px",
          backgroundColor: "#4CAF50",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        {loading ? "Đang chạy..." : "Chạy Decision Tree"}
      </button>

      {loading && (
        <div className="loading-indicator">
          <p>Đang xử lý dữ liệu, vui lòng chờ...</p>
          <div className="spinner"></div>
        </div>
      )}


{!loading && (
  <div>
        <Tabs style={{ marginTop: "30px" }}>
          <TabList>
            <Tab>Bước</Tab>
            <Tab>Kết quả</Tab>
          </TabList>

          <TabPanel>
          <div className="steps-container">
                <h2>Các bước xử lý</h2>
                <table className="steps-table">
                    <thead>
                        <tr>
                            <th className="header-cell">Số thứ tự</th>
                            <th className="header-cell">Mô tả</th>
                        </tr>
                    </thead>
                    <tbody>
                        {steps.map((step, index) => (
                            <tr key={index} className={index % 2 === 0 ? 'even-row' : 'odd-row'}>
                                <td className="cell">{index + 1}</td>
                                <td className="cell">{step}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
          </TabPanel>



      {!loading && treeOutput && (
        <div className="results" style={{ marginTop: "30px" }}>
          <h3 style={{ color: "#FF5722" }}>Kết quả Biểu Đồ</h3>
          <div
            dangerouslySetInnerHTML={{ __html: treeOutput }}
          />
        </div>
      )}
      
      

                </Tabs>

</div> 
    
)}
               
    
    </div>
  );
};

export default DecisionTreeGini;
