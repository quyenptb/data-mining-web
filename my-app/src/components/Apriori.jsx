import React, { useState } from "react";
import "./Apriori.css";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";



const Apriori = ({ fileContent }) => {
  const [minSupport, setMinSupport] = useState(0.1);
  const [minConf, setMinConf] = useState(0.1);
  const [frequentItemsets, setFrequentItemsets] = useState(null);
  const [maximalItemsets, setMaximalItemsets] = useState(null);
  const [rules, setRules] = useState(null);
  const [steps, setSteps] = useState([]);
  const [loading, setLoading] = useState(false);


  const handleAprioriClick = () => {
    if (!fileContent || fileContent.length === 0) {
      alert("Dữ liệu file trống! Hãy chắc chắn rằng bạn đã tải file.");
      return;
    }
  
    setLoading(true); // Hiển thị loading
    setFrequentItemsets(null); // Xóa dữ liệu cũ
    setMaximalItemsets(null);
    setSteps([]);
    setRules(null);
  
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
        min_support: minSupport,
        min_conf: minConf,
      };
  
      fetch("http://127.0.0.1:8000/myapp/apriori/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => response.text())
        .then((text) => {
          const sanitizedText = text.replace(/Infinity/g, "null");
          try {
            const data = JSON.parse(sanitizedText);
            if (data && data.rules_list) {
              setFrequentItemsets(data.frequent_itemsets || []);
              setMaximalItemsets(data.maximal_itemsets || []);
              setRules(data.rules_list || []);
              setSteps(data.steps || []);
            } else {
              console.error("Kết quả từ API không hợp lệ");
            }
          } catch (error) {
            console.error("Lỗi khi parse dữ liệu JSON:", error);
          }
        })
        .catch((error) => {
          console.error("Lỗi khi gửi yêu cầu Apriori:", error);
        })
        .finally(() => setLoading(false)); // Tắt loading sau khi xử lý xong
    }, 1500); // Giả lập chờ 3 giây
  };
  

  const formatDecimal = (value, decimals = 2) =>
    value ? value.toFixed(decimals) : "N/A";

  return (
    <div className="apriori-container" style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2 style={{ textAlign: "center", color: "#4CAF50" }}>Thuật toán Apriori</h2>
      <p style={{ textAlign: "justify", color: "#555" }}>
        Apriori là một thuật toán khai phá dữ liệu mạnh mẽ để tìm ra các mẫu kết
        hợp trong dữ liệu lớn. Thuật toán này giúp khám phá các quy tắc kết hợp
        giá trị trong các tập dữ liệu lớn một cách hiệu quả.
      </p>

      <div className="input-group" style={{ marginBottom: "20px" }}>
        <label htmlFor="min-support" style={{ marginRight: "10px" }}>
          Nhập giá trị <strong>min_support</strong>:
        </label>
        <input
          type="number"
          id="min-support"
          value={minSupport}
          onChange={(e) => setMinSupport(parseFloat(e.target.value))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
        <br></br>
        <label htmlFor="min-conf" style={{ marginRight: "10px", marginTop: "10px" }}>
          Nhập giá trị <strong>min_conf</strong>:
        </label>
        <input
          type="number"
          id="min-cof"
          value={minConf}
          onChange={(e) => setMinConf(parseFloat(e.target.value))}
          style={{ padding: "5px", borderRadius: "4px", border: "1px solid #ccc" }}
        />
      </div>

      <button
        onClick={handleAprioriClick}
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
        {loading ? "Đang chạy..." : "Chạy Apriori"}
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
      <div className="results" style={{ marginTop: "30px" }}>
        {frequentItemsets && frequentItemsets.length > 0 && (
          <div className="frequent-itemsets">
            <h3 style={{ color: "#FF5722" }}>1. Tập phổ biến</h3>
            <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: "20px" }}>
              <thead>
                <tr style={{ backgroundColor: "#f2f2f2" }}>
                  <th style={{ padding: "10px", border: "1px solid #ddd" }}>Tập phổ biến</th>
                  <th style={{ padding: "10px", border: "1px solid #ddd" }}>Độ phổ biến (Support)</th>
                </tr>
              </thead>
              <tbody>
                {frequentItemsets.map((itemset, index) => (
                  <tr key={index}>
                    <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                      {itemset.itemsets.join(", ")}
                    </td>
                    <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                      {formatDecimal(itemset.support)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {maximalItemsets && maximalItemsets.length > 0 && (
          <div className="maximal-itemsets">
            <h3 style={{ color: "#FF5722" }}>2. Tập phổ biến tối đại</h3>
            <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
              {maximalItemsets.map((itemset, index) => (
                <li key={index} style={{ marginBottom: "5px" }}>
                  {itemset.join(", ")}
                </li>
              ))}
            </ul>
          </div>
        )}

        {rules && rules.length > 0 && (
          <div className="association-rules">
            <h3 style={{ color: "#FF5722" }}>3. Luật kết hợp</h3>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ backgroundColor: "#f2f2f2" }}>
                  <th style={{ padding: "10px", border: "1px solid #ddd" }}>Antecedents</th>
                  <th style={{ padding: "10px", border: "1px solid #ddd" }}>Consequents</th>
                  <th style={{ padding: "10px", border: "1px solid #ddd" }}>Support</th>
                  <th style={{ padding: "10px", border: "1px solid #ddd" }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {rules.map((rule, index) => (
                  <tr key={index}>
                    <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                      {rule.antecedents.join(", ")}
                    </td>
                    <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                      {rule.consequents.join(", ")}
                    </td>
                    <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                      {formatDecimal(rule.support)}
                    </td>
                    <td style={{ padding: "10px", border: "1px solid #ddd" }}>
                      {formatDecimal(rule.confidence)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      </TabPanel>
                </Tabs>

      </div>
    )}
    </div>
    
    
  );
};

export default Apriori;
