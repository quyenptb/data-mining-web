import React, { createContext, useContext, useState } from "react";

export const EditedContentContext = createContext();

export const EditedContentProvider = ({ children }) => {
  const [editedContent, setEditedContent] = useState([]);

  return (
    <EditedContentContext.Provider value={{ editedContent, setEditedContent }}>
      {children}
    </EditedContentContext.Provider>
  );
};

export const useEditedContent = () => useContext(EditedContentContext);
