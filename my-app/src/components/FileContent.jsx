import React, { useState, useEffect } from "react";
import { useEditedContent } from './AppContext'
import * as XLSX from "xlsx";
import Papa from "papaparse";
import { saveAs } from "file-saver";
import "./FileContent.css";

const saveFileContent = (fileType, data) => {
  if (!data || data.length === 0) {
    console.error("Dữ liệu không hợp lệ hoặc rỗng:", data);
    alert("Dữ liệu không hợp lệ hoặc rỗng!");
    return;
  }

  if (!fileType) {
    console.error("Loại file không hợp lệ:", fileType);
    alert("Loại file không hợp lệ!");
    return;
  }

  let blob;
  try {
    switch (fileType) {
      case "csv":
        const csv = Papa.unparse(data);
        blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
        break;

      case "xlsx":
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
        blob = XLSX.write(wb, { bookType: "xlsx", type: "blob" });
        break;

      case "json":
        const json = JSON.stringify(data, null, 2);
        blob = new Blob([json], { type: "application/json;charset=utf-8" });
        break;

      case "txt":
        const txt = data.map(row => Object.values(row).join("\t")).join("\n");
        blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
        break;

      case "xml":
        const xml = `
          <root>
            ${data
              .map(
                row =>
                  `<item>${Object.entries(row)
                    .map(([key, value]) => `<${key}>${value}</${key}>`)
                    .join("")}</item>`
              )
              .join("")}
          </root>
        `;
        blob = new Blob([xml], { type: "application/xml;charset=utf-8" });
        break;

      default:
        console.error(`Loại file ${fileType} không được hỗ trợ!`);
        alert(`Loại file ${fileType} không được hỗ trợ!`);
        return;
    }
    saveAs(blob, `exported-file.${fileType}`);
  } catch (error) {
    console.error("Lỗi trong quá trình xử lý file:", error);
    alert("Đã xảy ra lỗi trong quá trình xử lý file!");
  }
};





const calculateStats = (data) => {
    const numberOfRows = data.length;
    const numberOfColumns = Object.keys(data[0] || {}).length;
  
    const emptyColumns = Object.keys(data[0] || {}).filter((colKey) =>
      data.every((row) => !row[colKey])
    ).length;
  
    const nonEmptyCells = data.reduce((acc, row) => {
      Object.values(row).forEach(value => {
        if (value !== null && value !== undefined && value !== '') acc++;
      });
      return acc;
    }, 0);
  
    const getMaxMin = (columnData) => {
      const numericData = columnData.filter(value => !isNaN(value));
      return numericData.length > 0 ? {
        max: Math.max(...numericData),
        min: Math.min(...numericData),
      } : { max: null, min: null };
    };
  
    const getAverage = (columnData) => {
      const numericData = columnData.filter(value => !isNaN(value));
      return numericData.length > 0 ? numericData.reduce((acc, value) => acc + value, 0) / numericData.length : null;
    };
  
    const getUniqueValues = (columnData) => {
      return new Set(columnData).size;
    };
  
    // Tính mode (giá trị phổ biến nhất)
    const getMode = (columnData) => {
      const frequency = {};
      let maxFreq = 0;
      let mode = null;
      columnData.forEach(value => {
        if (value !== undefined && value !== null && value !== "") {
          frequency[value] = (frequency[value] || 0) + 1;
          if (frequency[value] > maxFreq) {
            maxFreq = frequency[value];
            mode = value;
          }
        }
      });
      return mode;
    };
  
    const columnStats = Object.keys(data[0] || {}).reduce((acc, colKey) => {
      const columnData = data.map(row => row[colKey]);
      acc[colKey] = {
        uniqueValues: getUniqueValues(columnData),
        maxMin: getMaxMin(columnData),
        average: getAverage(columnData),
        mode: getMode(columnData),  // Thêm mode vào đây
      };
      return acc;
    }, {});
  
    return {
      numberOfRows,
      numberOfColumns,
      emptyColumns,
      nonEmptyCells,
      columnStats,
    };
  };
  


const FileContent = ({ fileContent, fileType }) => { 

  const { editedContent, setEditedContent } = useEditedContent();

  const [fileStats, setFileStats] = useState(null);
  const [history, setHistory] = useState([fileContent]);
  const [historyIndex, setHistoryIndex] = useState(0);

   // Initialize `editedContent` with `fileContent` on the first render or when `fileContent` changes
   useEffect(() => {
    setEditedContent(fileContent);
  }, [fileContent, setEditedContent]);

  // Calculate file stats when `editedContent` changes
  useEffect(() => {
    if (editedContent && editedContent.length > 0) {
      setFileStats(calculateStats(editedContent));
    }
  }, [editedContent]);

  
  

  const updateHistory = (newContent) => {
    const newHistory = [...history.slice(0, historyIndex + 1), newContent];
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleChange = (e, rowIndex, colKey) => {
    const newData = [...editedContent];
    newData[rowIndex][colKey] = e.target.value;
    setEditedContent(newData);
    updateHistory(newData);
  };

  const addColumn = () => {
    const newColumnKey = prompt("Nhập tên cột mới:");
    if (!newColumnKey) return;

    const updatedContent = editedContent.map((row) => ({
      ...row,
      [newColumnKey]: "",
    }));
    setEditedContent(updatedContent);
    updateHistory(updatedContent);
  };

  const deleteColumn = (colKey) => {
    const updatedContent = editedContent.map((row) => {
      const { [colKey]: _, ...rest } = row;
      return rest;
    });
    setEditedContent(updatedContent);
    updateHistory(updatedContent);
  };

  const addRow = () => {
    const newRow = {};
    Object.keys(editedContent[0] || {}).forEach((key) => {
      newRow[key] = "";
    });
    setEditedContent([...editedContent, newRow]);
    updateHistory([...editedContent, newRow]);
  };

  const deleteRow = (rowIndex) => {
    const updatedContent = editedContent.filter((_, index) => index !== rowIndex);
    setEditedContent(updatedContent);
    updateHistory(updatedContent);
  };

  const handleSave = () => {
    if (!fileType) {
      console.error("Loại file không được chọn:", fileType);
      alert("Vui lòng chọn loại file!");
      return;
    }
  
    if (!editedContent || editedContent.length === 0) {
      console.error("Dữ liệu không hợp lệ hoặc trống:", editedContent);
      alert("Dữ liệu không hợp lệ hoặc trống!");
      return;
    }
  
    saveFileContent(fileType, editedContent);
  };
  
  
  

  const searchAndReplace = (searchValue, replaceValue) => {
    const updatedContent = editedContent.map(row =>
      Object.keys(row).reduce((acc, key) => {
        acc[key] = row[key] === searchValue ? replaceValue : row[key];
        return acc;
      }, {})
    );
    setEditedContent(updatedContent);
    updateHistory(updatedContent);
  };

  const cleanData = () => {
    const cleanedContent = editedContent.filter(row =>
      Object.values(row).some(value => value !== "")
    );
    setEditedContent(cleanedContent);
    updateHistory(cleanedContent);
  };

  const importJSON = (jsonString) => {
    try {
      const jsonData = JSON.parse(jsonString);
      setEditedContent(jsonData);
      updateHistory(jsonData);
    } catch (error) {
      alert("Dữ liệu JSON không hợp lệ");
    }
  };

  const exportJSON = () => {
    const json = JSON.stringify(editedContent, null, 2);
    const blob = new Blob([json], { type: "application/json;charset=utf-8" });
    saveAs(blob, "data.json");
  };

  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setEditedContent(history[historyIndex - 1]);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setEditedContent(history[historyIndex + 1]);
    }
  };

  const fillEmptyCells = () => {
    if (!fileStats || !fileStats.columnStats) {
      alert("Thống kê chưa được tính toán!");
      return;
    }
  
    const updatedContent = editedContent.map((row) => {
      const newRow = { ...row };
      Object.keys(row).forEach((colKey) => {
        if (row[colKey] === "") {
          // Nếu cột là danh mục, dùng mode (giá trị phổ biến nhất)
          if (typeof row[colKey] === "string") {
            newRow[colKey] = fileStats.columnStats[colKey]?.mode || "";
          } else {
            // Nếu cột là số, dùng mean (trung bình)
            newRow[colKey] = fileStats.columnStats[colKey]?.average || "";
          }
        }
      });
      return newRow;
    });
  
    setEditedContent(updatedContent);
    updateHistory(updatedContent);
  };
  
  
  

  return (
    <div className="file-content">
      <h2>Trình chỉnh sửa file</h2>
      <div className="actions">
        <button onClick={addColumn}>Thêm cột</button>
        <button onClick={addRow}>Thêm dòng</button>
        <button onClick={handleSave}>Lưu lại</button>
        <button onClick={undo}>Hoàn tác</button>
        <button onClick={redo}>Làm lại</button>
        <button onClick={cleanData}>Làm sạch dữ liệu</button>
        <button onClick={fillEmptyCells}>Điền ô trống</button>
      </div>
      <div>
        {fileStats && (
          <div className="file-stats">
            <p><strong>Số dòng:</strong> {fileStats.numberOfRows}</p>
            <p><strong>Số cột:</strong> {fileStats.numberOfColumns}</p>
            <p><strong>Cột trống:</strong> {fileStats.emptyColumns}</p>
          </div>
        )}
      </div>
      <div className="table-wrapper">
        <table className="editable-table">
          <thead>
            <tr>
              {Object.keys(editedContent[0] || {}).map((key) => (
                <th className="editable-table-th" key={key}>
                  {key}
                  <button
                    className="delete-column"
                    onClick={() => deleteColumn(key)}
                    title="Xóa cột này"
                  >
                    &times;
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {editedContent.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {Object.entries(row).map(([colKey, value]) => (
                  <td key={colKey}>
                    <input
                      type="text"
                      value={value || ""}
                      onChange={(e) => handleChange(e, rowIndex, colKey)}
                    />
                  </td>
                ))}
                <td>
                  <button
                    className="delete-row"
                    onClick={() => deleteRow(rowIndex)}
                    title="Xóa dòng này"
                  >
                    &times;
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
    </div>
  );
};

export default FileContent;
