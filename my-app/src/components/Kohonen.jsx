import React, { useState } from "react";
import "./Kohonen.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";



const Kohonen = ({ fileContent }) => {
  const [gridSize, setGridSize] = useState([5, 5]); // Kích thước lưới mặc định
  const [learningRate, setLearningRate] = useState(0.5); // Tốc độ học mặc định
  const [numIterations, setNumIterations] = useState(100); // Số vòng lặp mặc định
  const [clusters, setClusters] = useState(null); // Kết quả các cụm
  const [bmuIndices, setBmuIndices] = useState(null); // Chỉ số BMU
  const [weights, setWeights] = useState(null); // Trọng số của các nút
  const [somImage, setSomImage] = useState(null); // Hình ảnh SOM
  const [loading, setLoading] = useState(false); // Trạng thái loading
  const [steps, setSteps] = useState([]);

  const handleKohonenClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }

    setLoading(true); // Hiển thị loading
    setClusters(null); // Xóa kết quả cũ
    setBmuIndices(null); // Xóa chỉ số BMU cũ
    setWeights(null); // Xóa trọng số cũ
    setSomImage(null); // Xóa hình ảnh SOM cũ
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
        data: sanitizedFileContent,
        grid_size: gridSize,
        learning_rate: learningRate,
        num_iterations: numIterations,
      };

      fetch("http://127.0.0.1:8000/myapp/kohonen/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data && data.clusters && data.som_image) {
            setClusters(data.clusters || []);
            setBmuIndices(data.bmu_indices || []);
            setWeights(data.weights || []);
            setSomImage(data.som_image || "");
            setSteps(data.steps || []);

          } else {
            console.error("Kết quả từ API không hợp lệ");
          }
        })
        .catch((error) => {
          console.error("Lỗi khi gửi yêu cầu Kohonen:", error);
        })
        .finally(() => setLoading(false)); // Tắt loading sau khi xử lý xong
    }, 1500); // Giả lập chờ 1.5 giây
  };

  const formatDecimal = (value, decimals = 2) =>
    value ? value.toFixed(decimals) : "N/A";

  return (
    <div className="kohonen-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Thuật toán Kohonen (SOM)</h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        Kohonen (Self-Organizing Map) là một thuật toán học không giám sát, được sử dụng để phân nhóm dữ liệu vào các cụm dựa trên sự tương đồng.
      </p>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="grid-size" style={{ marginRight: "10px" }}>
          Nhập kích thước lưới (grid_size):
        </label>
        <input
          type="text"
          id="grid-size"
          value={gridSize.join(", ")}
          onChange={(e) => setGridSize(e.target.value.split(",").map(Number))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="learning-rate" style={{ marginRight: "10px" }}>
          Nhập tốc độ học (learning_rate):
        </label>
        <input
          type="number"
          id="learning-rate"
          value={learningRate}
          onChange={(e) => setLearningRate(parseFloat(e.target.value))}
          step="0.01"
          min="0.01"
          max="1"
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="num-iterations" style={{ marginRight: "10px" }}>
          Nhập số vòng lặp (num_iterations):
        </label>
        <input
          type="number"
          id="num-iterations"
          value={numIterations}
          onChange={(e) => setNumIterations(parseInt(e.target.value))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <button
        onClick={handleKohonenClick}
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
        {loading ? "Đang chạy..." : "Chạy Kohonen"}
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


      {/* Hiển thị hình ảnh SOM nếu có */}
      {!loading && somImage && (
        <div>
          <h3>Kết quả huấn luyện SOM (Đồ họa)</h3>
          <img src={`data:image/png;base64,${somImage}`} alt="Self-Organizing Map (SOM)" />
        </div>
      )}
{/* Hiển thị kết quả cụm và BMU */}
{!loading && clusters && Object.keys(clusters).length > 0 && (
  <div className="results" style={{ marginTop: "30px" }}>
    <h3 style={{ color: "#FF5722" }}>Kết quả phân cụm</h3>
    <div className="clusters">
      {Object.entries(clusters).map(([bmuIdx, indices], idx) => {
        // Chuyển đổi bmuIdx từ định dạng chuỗi np.int64 về dạng tuple
        const bmuIdxKey = bmuIdx.replace(/np\.int64\((\d+)\)/g, "$1"); // Thay thế np.int64(x) bằng x
        const bmuIdxParsed = bmuIdxKey.split(","); // Tách thành mảng
        const bmuIdxString = `${bmuIdxParsed.join(", ")}`; // Tạo lại chuỗi cho bmuIdx

        return (
          <div key={idx} style={{ marginBottom: "20px" }}>
            <strong>BMU {bmuIdxString}:</strong>
            <ul>
              {indices.map((point, i) => (
                <li key={i}>
                  Điểm {point.name} ({point.coordinates.join(", ")})
                </li>
              ))}
            </ul>
          </div>
        );
      })}
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

export default Kohonen;
