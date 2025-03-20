import React, { useState } from "react";
import "./NaiveBayes.css";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { solarizedlight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Tooltip } from 'react-tooltip';


const NaiveBayes = ({ fileContent }) => {
  const [features, setFeatures] = useState("");
  const [target, setTarget] = useState("");
  const [query, setQuery] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [steps, setSteps] = useState(null); // Để lưu các bước
  const [loading, setLoading] = useState(false);


  const renderTable = (data, level = 0) => {
    if (typeof data !== "object" || data === null) {
      return <span>{JSON.stringify(data)}</span>;
    }
  
    return (
      <div className="custom-table-wrapper">
        <table className="custom-table">
          <thead>
            {level === 0 && (
              <tr>
                <th>Key</th>
                <th>Value</th>
              </tr>
            )}
          </thead>
          <tbody>
            {Object.entries(data).map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>
                  {typeof value === "object" && value !== null
                    ? renderTable(value, level + 1)
                    : JSON.stringify(value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  
  
  

  const handleNaiveBayesClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }

    if (!features || !target || Object.keys(query).length === 0) {
      alert("Hãy nhập đầy đủ thông tin Features, Target và Query!");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setSteps(null); // Reset các bước khi chạy lại thuật toán

    const requestBody = {
      fileContent: fileContent,
      features: features.split(",").map((f) => f.trim()), // Chuyển chuỗi Features thành danh sách
      target: target.trim(),
      query: query, // Dữ liệu Query đã được nhập
    };

    fetch("http://127.0.0.1:8000/myapp/naive_bayes/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data && data.prediction) {
          setPrediction(data.prediction);
          setSteps(data.steps); // Lưu các bước vào state
        } else {
          alert("Không nhận được kết quả dự đoán hợp lệ.");
        }
      })
      .catch((error) => {
        console.error("Lỗi khi gửi yêu cầu Naive Bayes:", error);
      })
      .finally(() => setLoading(false));
  };

  const handleQueryChange = (key, value) => {
    setQuery((prevQuery) => ({
      ...prevQuery,
      [key]: value,
    }));
  };

   // Tạo câu kết luận
   const generateConclusion = () => {
    const queryString = Object.entries(query)
      .map(([key, value]) => `${key} is ${value}`.toLowerCase())
      .join(", ");
    return `If the ${queryString}
    then you should choose ${prediction.toLowerCase()} to ${target.toLowerCase()}.`;
  };

  return (

        <div className="naive-bayes-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
          <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Thuật toán Naive Bayes</h2>
          <p style={{ textAlign: "justify", color: "#555" }}>
            Naive Bayes là một thuật toán phân loại dựa trên xác suất, thường được sử dụng để phân tích văn bản, dự đoán, và phân loại dữ liệu.
          </p>
    
          {/* Nhập Features */}
          <div className="input-group" style={{ marginBottom: "20px" }}>
            <label htmlFor="features" style={{ display: "block", marginBottom: "10px" }}>
              Nhập các Features (phân cách bởi dấu phẩy):
            </label>
            <input
              type="text"
              id="features"
              value={features}
              onChange={(e) => setFeatures(e.target.value)}
              placeholder="Ví dụ: Outlook, Temperature, Humidity, Wind"
              style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
              data-tip="Nhập các đặc tính bạn muốn sử dụng để dự đoán, phân cách bằng dấu phẩy."
            />
            <Tooltip place="top" type="dark" effect="solid" />
          </div>
    
          {/* Nhập Target */}
          <div className="input-group" style={{ marginBottom: "20px" }}>
            <label htmlFor="target" style={{ display: "block", marginBottom: "10px" }}>
              Nhập Target (cột mục tiêu):
            </label>
            <input
              type="text"
              id="target"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="Ví dụ: Play"
              style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
              data-tip="Nhập cột mục tiêu (target) mà bạn muốn dự đoán, ví dụ như 'Play'."
            />
            <Tooltip place="top" type="dark" effect="solid" />
          </div>
    
          {/* Nhập Query */}
          <div className="input-group" style={{ marginBottom: "20px" }}>
            <label htmlFor="query" style={{ display: "block", marginBottom: "10px" }}>
              Nhập Query (đối tượng cần dự đoán):
            </label>
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
              {features.split(",").map((feature, index) => (
                <div key={index} style={{ flex: "1 1 100%" }}>
                  <label htmlFor={`query-${feature.trim()}`} style={{ marginRight: "10px" }}>
                    {feature.trim()}:
                  </label>
                  <input
                    type="text"
                    id={`query-${feature.trim()}`}
                    onChange={(e) => handleQueryChange(feature.trim(), e.target.value)}
                    placeholder={`Giá trị cho ${feature.trim()}`}
                    style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc", width: "100%" }}
                    data-tip={`Nhập giá trị cho ${feature.trim()} trong query để dự đoán.`}
                  />
                  <Tooltip place="top" type="dark" effect="solid" />
                </div>
              ))}
            </div>
          </div>

      {/* Nút thực hiện Naive Bayes */}
      <button
        onClick={handleNaiveBayesClick}
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
        {loading ? "Đang chạy..." : "Chạy Naive Bayes"}
      </button>

      {/* Hiển thị quá trình tính toán */}
      {loading && (
        <div className="loading-indicator">
          <p>Đang xử lý dữ liệu, vui lòng chờ...</p>
          <div className="spinner"></div>
        </div>
      )}

      {/* Kết quả bước 1: Prior Probabilities */}
{steps && !loading && (
  <div
    className="naive-bayes__steps"
    style={{
      display: "flex",
      flexDirection: "column",
      gap: "20px",
      marginTop: "30px",
    }}
  >
    <>
      <div className="naive-bayes__step-item">
        <h3 style={{ color: "#FF5722" }}>Bước 1: Xác suất tiên nghiệm (Prior Probabilities)</h3>
        {renderTable(steps.prior_probabilities)}
      </div>

      <div className="naive-bayes__step-item">
        <h3 style={{ color: "#FF5722" }}>Bước 2: Xác suất có điều kiện (Conditional Probabilities)</h3>
        {renderTable(steps.conditional_probabilities)}
      </div>

      <div className="naive-bayes__step-item">
        <h3 style={{ color: "#FF5722" }}>Bước 3: Xác suất hậu nghiệm (Posterior Probabilities)</h3>
        {renderTable(steps.posterior_probabilities)}
      </div>

      <div className="naive-bayes__step-item">
        <h3 style={{ color: "#FF5722" }}>Bước 4: Dự đoán kết quả</h3>
        <p>
          <strong>Kết quả:</strong> {prediction}
          <p>{generateConclusion()}</p>

        </p>
      </div>
    </>
  </div>
)}

    </div>
  );
};

export default NaiveBayes;
