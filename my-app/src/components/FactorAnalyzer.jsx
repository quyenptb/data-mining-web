import React, { useState } from "react";
import "./FactorAnalyzer.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";


const FactorAnalyzer = ({ fileContent }) => {
  const [numFactors, setNumFactors] = useState(3); // Số lượng nhân tố mặc định
  const [loadings, setLoadings] = useState(null); // Kết quả tải nhân tố
  const [clusters, setClusters] = useState(null); // Kết quả cụm
  const [loading, setLoading] = useState(false);
  const [graph, setGraph] = useState(null); // Thêm trạng thái cho đồ họa
  const [steps, setSteps] = useState([]);

  const handleFactorAnalyzerClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }
  
    setLoading(true); // Hiển thị loading
    setLoadings(null); // Xóa kết quả cũ
    setClusters(null); // Xóa cụm cũ
    setSteps([]);
  
    // Giả lập thời gian chờ (1.5 giây) trước khi gửi yêu cầu đến API
    setTimeout(() => {
      // Loại bỏ các giá trị không hợp lệ trong fileContent
      const sanitizedFileContent = fileContent.map((item) => {
        const sanitizedItem = { ...item };
        for (const key in sanitizedItem) {
          if (
            sanitizedItem[key] === Infinity ||
            sanitizedItem[key] === -Infinity ||
            Number.isNaN(sanitizedItem[key])
          ) {
            sanitizedItem[key] = null; // Thay thế giá trị không hợp lệ bằng null
          }
        }
        return sanitizedItem;
      });
  
      const requestBody = {
        points: sanitizedFileContent,
        num_factors: numFactors,
      };
  
      fetch("http://127.0.0.1:8000/myapp/factor-analyzer/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.text(); // Đọc phản hồi dưới dạng văn bản thô
        })
        .then((rawData) => {
          // Xử lý JSON không hợp lệ bằng cách thay thế NaN, Infinity
          const sanitizedResponse = rawData.replace(/NaN|Infinity|-Infinity/g, "null");
          const data = JSON.parse(sanitizedResponse);
  
          if (data && data.loadings && data.clusters) {
            setLoadings(data.loadings || []);
            setClusters(data.clusters || []);
            setGraph(data.graph); // Nhận và lưu trữ hình ảnh đồ họa
            setSteps(data.steps || []);

          } else {
            console.error("Kết quả từ API không hợp lệ", data);
          }
        })
        .catch((error) => {
          console.error("Lỗi khi gửi yêu cầu FactorAnalyzer:", error);
        })
        .finally(() => setLoading(false)); // Tắt loading sau khi xử lý xong
    }, 1500); // Giả lập chờ 1.5 giây
  };
  



  const formatDecimal = (value, decimals = 2) =>
    value ? value.toFixed(decimals) : "N/A";

  return (
    <div
      className="factor-analyzer-container"
      style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}
    >
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>
        Phân tích nhân tố
      </h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        Phân tích nhân tố là một kỹ thuật thống kê được sử dụng để phân tích cấu trúc của một tập dữ liệu lớn, xác định số lượng nhân tố và tải nhân tố cho mỗi biến.
      </p>
  
      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="num-factors" style={{ marginRight: "10px" }}>
          Nhập số nhân tố <strong>num_factors</strong>:
        </label>
        <input
          type="number"
          id="num-factors"
          value={numFactors}
          onChange={(e) => setNumFactors(parseInt(e.target.value))}
          style={{
            padding: "5px",
            borderRadius: "4px",
            border: "1px solid #ccc",
          }}
        />
      </div>
  
      <button
        onClick={handleFactorAnalyzerClick}
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
        {loading ? "Đang chạy..." : "Chạy FactorAnalyzer"}
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
<TabPanel>

  
      {/* Hiển thị đồ họa nếu có */}
      {!loading && graph && (
        <div style={{ marginTop: "20px" }}>
          <h3 style={{ color: "#2196F3" }}>Kết quả phân tích (Đồ họa)</h3>
          <img
            src={`data:image/png;base64,${graph}`}
            alt="Factor Analysis"
            style={{ width: "100%", maxWidth: "600px", margin: "auto" }}
          />
        </div>
      )}

      {
        console.log("Day là loading và cluster", loadings, clusters)
      }
  
      {!loading && loadings && Object.keys(loadings).length > 0 && (
        <div className="results" style={{ marginTop: "30px" }}>
          <div className="loadings">
            <h3 style={{ color: "#FF5722" }}>Ma trận tải nhân tố</h3>
            <table
              border="1"
              cellPadding="5"
              style={{
                borderCollapse: "collapse",
                width: "100%",
                textAlign: "center",
              }}
            >
              <thead>
                <tr>
                  <th>Variable</th>
                  {Object.keys(loadings).map((factor, index) => (
                    <th key={index}>{factor}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.keys(loadings["Factor 1"]).map((_, varIndex) => (
                  <tr key={varIndex}>
                    <td>Variable {varIndex + 1}</td>
                    {Object.keys(loadings).map((factor, factorIndex) => (
                      <td key={factorIndex}>
                        {loadings[factor][varIndex].toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
  
          {!loading && clusters && Object.keys(clusters).length > 0 && (
  <div className="clusters" style={{ marginTop: "20px" }}>
    <h3 style={{ color: "#FF5722" }}>Kết quả cụm</h3>
    {Object.keys(clusters).map((factor, index) => (
      <div key={index} style={{ marginBottom: "10px" }}>
        <strong>{factor}:</strong>
        <ul style={{ paddingLeft: "20px" }}>
          {clusters[factor].map((variable, varIndex) => (
            <li key={varIndex}>{variable}</li>
          ))}
        </ul>
      </div>
    ))}
  </div>
)}

        </div>
      )}


      </TabPanel>

                </Tabs>

</div> 
    
)}
               
    
    </div>
  );
};
  export default FactorAnalyzer;
  