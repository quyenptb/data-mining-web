import React, { useState } from "react";
import axios from "axios";
import "./RoughSet.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

const RoughSet = ({ fileContent }) => {
  const [equivalenceClasses, setEquivalenceClasses] = useState(null);
  const [lowerApproximation, setLowerApproximation] = useState(null);
  const [upperApproximation, setUpperApproximation] = useState(null);
  const [gammaB_C, setGammaB_C] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [discernibilityMatrix, setDiscernibilityMatrix] = useState(null);
  const [simplifiedFormula, setSimplifiedFormula] = useState(null);
  const [loading, setLoading] = useState(false);
  const [matrixData, setMatrixData] = useState(null);
  const [isOn, setIsOn] = useState("")
  const [inputX, setInputX] = useState([]);
  const [inputB, setInputB] = useState([]);
  const [inputC, setInputC] = useState("");
  const [inputAttributes, setInputAttributes] = useState([]);
  const [steps, setSteps] = useState([]);


  const [selectedAction, setSelectedAction] = useState("");

  const fetchRoughSetData = async (action) => {
    setIsOn(action)
    setLoading(true);
    setSelectedAction(action);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/myapp/rough_set/",
        {
          action: action,
          X: inputX,
          B: inputB,
          C: inputC,
          attributes: inputAttributes,          
          //X: ["O1", "O3", "O4"],
          //B: ["Troi", "Gio"],
          //C: "Ketqua",
          //attributes: ['Troi', 'Gio', 'Apsuat'],           
          df2: fileContent
        },
        {
          headers: {
            "Content-Type": "application/json",
          }
        }
      );

      const { lower_approximation, upper_approximation, accuracy, equivalence_classes, dependency_degree, simplified_formula } = response.data;
      setSteps(response.data.steps || []);
      

      // Cập nhật kết quả theo hành động đã chọn
    if (action === "approximation") {
      setEquivalenceClasses(equivalence_classes);
      setLowerApproximation(lower_approximation);
      setUpperApproximation(upper_approximation);
      setAccuracy(accuracy);
    } else if (action === "dependency") {
      setGammaB_C(dependency_degree);
    } else if (action === "discernibility") {
      const rawData = response.data;
      const fixedDataString = rawData.replace(/NaN/g, "null");
      const parsedData = JSON.parse(fixedDataString);

      const discernibilityMatrix = parsedData.discernibility_matrix;

      if (!discernibilityMatrix) {
        alert("Dữ liệu trả về không chứa 'discernibility_matrix'.");
        return;
      }

      const data = Object.entries(discernibilityMatrix).map(([rowKey, rowValue]) => ({
        row: rowKey,
        columns: Object.entries(rowValue).map(([colKey, colValue]) => ({
          column: colKey,
          value: colValue.join(", "),
        })),
      }));

      setMatrixData(data);

      setDiscernibilityMatrix(data);
    } else if (action === "reducts") {
      setSimplifiedFormula(simplified_formula);
    }

      //setEquivalenceClasses(equivalence_classes);
      //setLowerApproximation(lower_approximation);
      //setUpperApproximation(upper_approximation);
      //setGammaB_C(dependency_degree);
      //setAccuracy(accuracy);
      //setDiscernibilityMatrix(discernibility_matrix);
      //setSimplifiedFormula(simplified_formula);
    } catch (error) {
      console.error("Có lỗi xảy ra khi tính toán Rough Set:", error);
      alert("Đã có lỗi xảy ra, vui lòng thử lại.");
    } finally {
      setLoading(false);
    }
  };

  const handleApproximation = () => fetchRoughSetData("approximation");
  const handleDependency = () => fetchRoughSetData("dependency");
  const handleDiscernibility = () => fetchRoughSetData("discernibility");
  const handleReducts = () => fetchRoughSetData("reducts");

  return (
    <div className="roughset-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Thuật toán Rough Set</h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        Rough Set là một phương pháp khai phá dữ liệu để phân tích các tập dữ liệu không rõ ràng, tìm ra các mối quan hệ giữa các thuộc tính và đánh giá sự phụ thuộc giữa các thuộc tính.
      </p>

      <div style={{ marginBottom: "20px" }}>
  <h3 style={{ color: "#4CAF50" }}>Nhập dữ liệu</h3>
  <div style={{ marginBottom: "10px" }}>
    <label style={{ marginRight: "10px" }}>Tập X:</label>
    <input
      type="text"
      placeholder="e.g., O1, O3, O4"
      value={inputX}
      onChange={(e) => setInputX(e.target.value.split(",").map((item) => item.trim()))}
      style={{ padding: "5px", width: "50%" }}
    />
  </div>
  <div style={{ marginBottom: "10px" }}>
    <label style={{ marginRight: "10px" }}>Tập B:</label>
    <input
      type="text"
      placeholder="e.g., Troi, Gio"
      value={inputB}
      onChange={(e) => setInputB(e.target.value.split(",").map((item) => item.trim()))}
      style={{ padding: "5px", width: "50%" }}
    />
  </div>
  <div style={{ marginBottom: "10px" }}>
    <label style={{ marginRight: "10px" }}>Thuộc tính C:</label>
    <input
      type="text"
      placeholder="e.g., Ketqua"
      value={inputC}
      onChange={(e) => setInputC(e.target.value)}
      style={{ padding: "5px", width: "50%" }}
    />
  </div>
  <div style={{ marginBottom: "10px" }}>
    <label style={{ marginRight: "10px" }}>Attributes:</label>
    <input
      type="text"
      placeholder="e.g., Troi, Gio, Apsuat"
      value={inputAttributes}
      onChange={(e) => setInputAttributes(e.target.value.split(",").map((item) => item.trim()))}
      style={{ padding: "5px", width: "50%" }}
    />
  </div>
</div>

      {/* Các nút cho các hành động */}
      <div className="action-buttons" style={{ textAlign: "center", marginBottom: "20px" }}>
        {["approximation", "dependency", "discernibility", "reducts"].map(action => (
          <button
            key={action}
            onClick={() => {
              if (action === "approximation") handleApproximation();
              if (action === "dependency") handleDependency();
              if (action === "discernibility") handleDiscernibility();
              if (action === "reducts") handleReducts();
            }}
            disabled={loading}
            style={{
              padding: "10px 20px",
              backgroundColor: selectedAction === action ? "#FF5722" : "#4CAF50",
              color: "#fff",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
              margin: "5px",
            }}
          >
            {loading && selectedAction === action ? "Đang chạy..." : action.charAt(0).toUpperCase() + action.slice(1)}
          </button>
        ))}
      </div>

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

      {!loading && (
        <div className="results" style={{ marginTop: "30px" }}>
          {equivalenceClasses && selectedAction == 'approximation' && (
      <div className="lower-approximation">
        <h3 style={{ color: "#FF5722" }}>1. Các lớp tương đương</h3>
        {/* Convert object to a list */}
        <div>
          {Object.entries(equivalenceClasses).map(([key, value]) => (
            <p key={key}>{key}: {value}</p>
          ))}
        </div>
      </div>
    )}

          {lowerApproximation && selectedAction == 'approximation' && (
            <div className="lower-approximation">
              <h3 style={{ color: "#FF5722" }}>2. Xấp xỉ dưới</h3>
              <p>{lowerApproximation.join(", ")}</p>
            </div>
          )}

          {upperApproximation && selectedAction == 'approximation' && (
            <div className="upper-approximation">
              <h3 style={{ color: "#FF5722" }}>3. Xấp xỉ trên</h3>
              <p>{upperApproximation.join(", ")}</p>
            </div>
          )}

          {accuracy !== null && selectedAction == 'approximation' && (
            <div className="upper-approximation">
              <h3 style={{ color: "#FF5722" }}>4. Độ chính xác</h3>
              <p>{accuracy}</p>
            </div>
          )}

          {gammaB_C !== null && selectedAction == 'dependency' &&  (
            <div className="dependence">
              <h3 style={{ color: "#FF5722" }}>5. Mức độ phụ thuộc</h3>
              <p>{gammaB_C}</p>
            </div>
          )}

{matrixData && selectedAction == 'discernibility' && (
  <div>
    <h3 style={{ color: "#FF5722" }}>6. Discernibility Matrix</h3>
    <table className="styled-table">
      <thead>
        <tr>
          <th>Row</th>
          {[...Array(matrixData[0].columns.length)].map((_, colIndex) => (
            <th key={colIndex}>Column {colIndex + 1}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrixData.map((row, rowIndex) => (
          <tr key={rowIndex}>
            <td>Row {rowIndex + 1}</td>
            {row.columns.map((col, colIndex) => (
              (colIndex <= rowIndex) ? (
                <td key={colIndex}>{JSON.stringify(col.value)}</td>
              ) : (
                <td key={colIndex}></td>
              )
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}


          {simplifiedFormula && selectedAction == 'reducts' && (
            <div className="simplifiedFormula">
              <h3 style={{ color: "#FF5722" }}>7. Tập thô</h3>
              <p>{simplifiedFormula}</p>
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

export default RoughSet;
