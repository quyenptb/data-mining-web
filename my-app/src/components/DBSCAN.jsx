import React, { useState } from "react";
import "./DBSCAN.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";



const DBSCAN = ({ fileContent }) => {
  const [eps, setEps] = useState(0.5); // Khoảng cách tối đa
  const [minSamples, setMinSamples] = useState(5); // Số lượng mẫu tối thiểu
  const [clusters, setClusters] = useState(null); // Kết quả cụm
  const [loading, setLoading] = useState(false);
  const [graph, setGraph] = useState(null); // Trạng thái cho đồ họa
  const [steps, setSteps] = useState([]);
  

  const handleDBSCANClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }

    setLoading(true); // Hiển thị loading
    setClusters(null); // Xóa kết quả cũ
    setSteps([]);

    // Giả lập thời gian chờ (3 giây) trước khi gửi yêu cầu đến API
    setTimeout(() => {
      const sanitizedFileContent = fileContent.map((item) => {
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
        points: sanitizedFileContent,
        eps: eps,
        min_samples: minSamples,
      };

      fetch("http://127.0.0.1:8000/myapp/dbscan/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data && data.clusters) {
            setClusters(data.clusters || []);
            setGraph(data.graph); // Nhận và lưu trữ hình ảnh đồ họa
            setSteps(data.steps || []);
          } else {
            console.error("Kết quả từ API không hợp lệ");
          }
        })
        .catch((error) => {
          console.error("Lỗi khi gửi yêu cầu DBSCAN:", error);
        })
        .finally(() => setLoading(false)); // Tắt loading sau khi xử lý xong
    }, 1500); // Giả lập chờ 1.5 giây
  };

  const formatDecimal = (value, decimals = 2) =>
    value ? value.toFixed(decimals) : "N/A";

  return (
    <div className="dbscan-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Thuật toán DBSCAN</h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        DBSCAN là thuật toán phân cụm dữ liệu dựa trên mật độ, cho phép phát hiện các cụm có hình dạng bất kỳ và phân loại các điểm nhiễu.
      </p>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="eps" style={{ marginRight: "10px" }}>
          Nhập khoảng cách tối đa <strong>eps</strong>:
        </label>
        <input
          type="number"
          id="eps"
          value={eps}
          onChange={(e) => setEps(parseFloat(e.target.value))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="min-samples" style={{ marginRight: "10px" }}>
          Nhập số lượng mẫu tối thiểu <strong>min_samples</strong>:
        </label>
        <input
          type="number"
          id="min-samples"
          value={minSamples}
          onChange={(e) => setMinSamples(parseInt(e.target.value))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <button
        onClick={handleDBSCANClick}
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
        {loading ? "Đang chạy..." : "Chạy DBSCAN"}
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
        <div>
          <h3>Kết quả phân cụm (Đồ họa)</h3>
          <img src={`data:image/png;base64,${graph}`} alt="DBSCAN Clustering" />
        </div>
      )}

{!loading && clusters && clusters.length > 0 && (
        <div className="results" style={{ marginTop: "30px" }}>
          <h3 style={{ color: "#FF5722" }}>Kết quả phân cụm</h3>
          {clusters.map((cluster, index) => (
            <div key={index}>
              <ul>
                {Object.entries(cluster).map(([clusterName, points]) => (
                  <li key={clusterName}>
                    <strong>{clusterName}:</strong>
                    <ul>
                      {points.map((point, i) => (
                        <li key={i}>
                          {point.name || "Điểm không tên"} ({formatDecimal(point.coordinates[0])}, {formatDecimal(point.coordinates[1])})
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

</TabPanel>
                </Tabs> 
                </div>

)} </div> 
    
  );
  
};

export default DBSCAN;