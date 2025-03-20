import React, { useState } from "react";
import "./KMeans.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";



const KMeans = ({ fileContent }) => {
  const [numClusters, setNumClusters] = useState(3); // Số lượng cụm mặc định
  const [clusters, setClusters] = useState(null); // Kết quả cụm
  const [centroids, setCentroids] = useState(null);
  const [loading, setLoading] = useState(false);
  const [graph, setGraph] = useState(null);  // Thêm trạng thái cho đồ họa
  const [steps, setSteps] = useState([]);


  const handleKMeansClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }

    setLoading(true); // Hiển thị loading
    setClusters(null); // Xóa kết quả cũ
    setSteps([]);

    // Giả lập thời gian chờ (3 giây) trước khi gửi yêu cầu đến API
    setTimeout(() => {
      // Chuyển đổi dữ liệu và gửi yêu cầu
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
        num_clusters: numClusters,
      };

      fetch("http://127.0.0.1:8000/myapp/kmeans/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data && data.clusters && data.centroids) {
            setClusters(data.clusters || []);
            setCentroids(data.centroids || []);
            setGraph(data.graph);  // Nhận và lưu trữ hình ảnh đồ họa
            setSteps(data.steps || []);


          } else {
            console.error("Kết quả từ API không hợp lệ");
          }
        })
        .catch((error) => {
          console.error("Lỗi khi gửi yêu cầu KMeans:", error);
        })
        .finally(() => setLoading(false)); // Tắt loading sau khi xử lý xong
    }, 1500); // Giả lập chờ 1.5 giây
  };

  const formatDecimal = (value, decimals = 2) =>
    value ? value.toFixed(decimals) : "N/A";

  return (
    <div className="kmeans-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Thuật toán KMeans</h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        KMeans là thuật toán phân cụm dữ liệu phổ biến, được sử dụng để phân chia các đối tượng vào các nhóm sao cho các đối tượng trong cùng một nhóm có sự tương đồng cao nhất.
      </p>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="num-clusters" style={{ marginRight: "10px" }}>
          Nhập số cụm <strong>num_clusters</strong>:
        </label>
        <input
          type="number"
          id="num-clusters"
          value={numClusters}
          onChange={(e) => setNumClusters(parseInt(e.target.value))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <button
        onClick={handleKMeansClick}
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
        {loading ? "Đang chạy..." : "Chạy KMeans"}
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
          <img src={`data:image/png;base64,${graph}`} alt="KMeans Clustering" />
        </div>
      )}

      {!loading && clusters && clusters.length > 0 && (
        <div className="results" style={{ marginTop: "30px" }}>
          <div className="clusters">
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
                            {point.name} ({formatDecimal(point.coordinates[0])}, {formatDecimal(point.coordinates[1])})
                          </li>
                        ))}
                      </ul>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <div className="centroids">
            <h3 style={{ color: "#FF5722" }}>Các centroid</h3>
            <ul>
              {centroids.map((centroid, index) => (
                <li key={index}>
                  Centroid {index + 1}: ({formatDecimal(centroid[0])}, {formatDecimal(centroid[1])})
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      </TabPanel>

</Tabs>

</div> 

)}


</div>
);
};

export default KMeans;
