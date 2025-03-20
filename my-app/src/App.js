import React from "react";
import Navbar from "./components/Navbar";
import GuideSection from "./components/GuideSection";
import LogoSection from "./components/LogoSection";
import Introduction from "./components/Introduction";

const App = () => {
  return (
    <div>
      <Navbar />
      <main>
        <GuideSection />
        <LogoSection />
      </main>
    </div>
  );
};

export default App;