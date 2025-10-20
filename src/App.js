import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import {
  Layout,
  Card,
  Row,
  Col,
  Typography,
  Statistic,
  Tag,
  Empty,
} from 'antd';
import {
  FileImageOutlined,
  MedicineBoxOutlined,
  HistoryOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

// 导入新的模块化组件
import { UploadImage, HeatmapDisplay } from './features/upload';
import { ReportGenerator, ReportDisplay } from './features/llava-report';
import { KnowledgeGraph } from './features/third-party-api';
import AnalysisHistory from './AnalysisHistory';
import './App.css';

const { Header, Content } = Layout;
const { Title } = Typography;

function Dashboard() {
  const navigate = useNavigate();

  // 由 UploadImage 上抛
  const [selectedFile, setSelectedFile] = useState(null);      // 可供知识图谱使用（若需要）
  const [analysisResults, setAnalysisResults] = useState(null);// 右侧“AI 分析结果”条件渲染依据
  const [generatedReport, setGeneratedReport] = useState(null);

  // 仅用于点击“查看历史”
  const handleHistoryClick = () => navigate('/history');

  return (
    <Layout style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)' }}>
      <Header
        style={{
          backgroundColor: 'white',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          borderBottom: '4px solid #3b82f6',
          padding: '0 24px'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '64px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <MedicineBoxOutlined style={{ fontSize: '32px', color: '#3b82f6' }} />
            <Title level={2} style={{ margin: 0, color: '#1e40af' }}>
              Medical AI Dashboard
            </Title>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <Tag color="blue" style={{ padding: '4px 16px', fontSize: '14px' }}>
              Healthcare Analytics
            </Tag>
          </div>
        </div>
      </Header>

      <Content style={{ padding: '24px' }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
          {/* 顶部统计卡片 */}
          <Row gutter={16} style={{ marginBottom: '24px' }}>
            <Col span={6}>
              <Card
                className="stat-card"
                style={{
                  textAlign: 'center',
                  background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                  color: 'white',
                  border: 'none',
                  cursor: 'default'
                }}
              >
                <Statistic
                  title={<span style={{ color: 'white' }}>Total Scans</span>}
                  value={1234}
                  prefix={<FileImageOutlined />}
                  valueStyle={{ color: '#fff' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card
                className="stat-card"
                style={{
                  textAlign: 'center',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: 'white',
                  border: 'none',
                  cursor: 'default'
                }}
              >
                <Statistic
                  title={<span style={{ color: 'white' }}>Normal Cases</span>}
                  value={89.7}
                  precision={1}
                  suffix={<span style={{ color: 'white' }}>%</span>}
                  prefix={<CheckCircleOutlined />}
                  valueStyle={{ color: '#fff' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card
                className="stat-card"
                style={{
                  textAlign: 'center',
                  background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                  color: 'white',
                  border: 'none',
                  cursor: 'default'
                }}
              >
                <Statistic
                  title={<span style={{ color: 'white' }}>Abnormal Cases</span>}
                  value={10.3}
                  precision={1}
                  suffix={<span style={{ color: 'white' }}>%</span>}
                  prefix={<ExclamationCircleOutlined />}
                  valueStyle={{ color: '#fff' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card
                className="stat-card"
                onClick={handleHistoryClick}
                style={{
                  textAlign: 'center',
                  background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
                  color: 'white',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'transform 0.2s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
              >
                <Statistic
                  title={<span style={{ color: 'white' }}>Analysis History</span>}
                  value="View All"
                  prefix={<HistoryOutlined />}
                  valueStyle={{ color: '#fff', fontSize: '16px' }}
                />
              </Card>
            </Col>
          </Row>

          {/* 主体：左上传 / 右知识图谱 + 结果（结果默认隐藏） */}
          <Row gutter={24}>
            {/* 左侧列 - 上传与分析 */}
            <Col span={7}>
              <UploadImage
                // 当用户选择文件时（新版 UploadImage 内部可以在 onChange 时上抛，若未上抛可忽略）
                onFileChange={(f) => setSelectedFile(f)}
                // 子组件调用接口后把结果抛上来
                onAnalysisFinished={(res) => setAnalysisResults(res?.success ? res : null)}
                // 兼容另一命名
                onResults={(res) => setAnalysisResults(res?.success ? res : null)}
              />
            </Col>

            {/* 右侧列 - 知识图谱 +（条件）AI 分析结果 */}
            <Col span={17}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', height: '100%' }}>
                {/* 医学知识图谱（保留原有卡片内容） */}
                <Card title="Medical Knowledge Graph">
                  {/* 如果你这里有真实的知识图谱可视化，就替换掉 Empty */}
                  <KnowledgeGraph analysisResults={analysisResults} selectedFile={selectedFile} />
                  {!analysisResults && !selectedFile && (
                    <Empty description="Upload a CT scan to view related medical knowledge graph" />
                  )}
                </Card>

                {/* ★ 新增：AI 分析结果（仅分析完成后显示） */}
                {analysisResults ? (
                  <Card title="AI 分析结果">
                    <HeatmapDisplay analysisResults={analysisResults} isAnalyzing={false} />
                  </Card>
                ) : null}
              </div>
            </Col>
          </Row>

          {/* LLaVA 报告生成模块（仅分析后出现） */}
          {analysisResults && (
            <Row gutter={24} style={{ marginTop: '24px' }}>
              <Col span={12}>
                <ReportGenerator
                  analysisData={analysisResults}
                  onReportGenerated={(report) => setGeneratedReport(report)}
                />
              </Col>
              <Col span={12}>
                <ReportDisplay report={generatedReport} />
              </Col>
            </Row>
          )}
        </div>
      </Content>
    </Layout>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/history" element={<AnalysisHistory />} />
      </Routes>
    </Router>
  );
}

export default App;
