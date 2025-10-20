// upload/components/UploadImage.jsx
import React, { useState } from 'react';
import { Upload, Card, message, Space, Typography, InputNumber, Switch, Button } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { uploadAndAnalyzeImage, validateFileType, validateFileSize, getFilePreviewUrl } from '../api';

const { Dragger } = Upload;
const { Text } = Typography;

export default function UploadImage(props) {
  const { onAnalyze, onAnalysisFinished, onResults } = props;

  const [file, setFile] = useState(null);
  const [fileList, setFileList] = useState([]);
  const [generateHeatmap, setGenerateHeatmap] = useState(true);
  const [threshold, setThreshold] = useState(0.5);
  const [topK, setTopK] = useState(null);
  const [loading, setLoading] = useState(false);

  const beforeUpload = (f) => {
    if (!validateFileType(f)) {
      message.error('仅支持 .dcm .jpg .jpeg .png');
      return Upload.LIST_IGNORE || false;
    }
    if (!validateFileSize(f)) {
      message.error('文件不能超过 50MB');
      return Upload.LIST_IGNORE || false;
    }
    return false;
  };

  const onChange = (info) => {
    const latest = info.file;
    const raw = latest?.originFileObj || latest;
    if (!raw || !validateFileType(raw) || !validateFileSize(raw)) {
      setFile(null);
      setFileList([]);
      return;
    }
    setFile(raw);
    setFileList([{ ...latest, status: 'done' }]);
  };

  const onRemove = () => {
    setFile(null);
    setFileList([]);
    return true;
  };

  const handleAnalyze = async () => {
    if (!file) {
      message.warning('请先选择文件');
      return;
    }
    const options = { generate_heatmap: generateHeatmap, threshold, alpha: 0.45, return_top_k: topK };

    // 父组件自定义（优先）
    if (typeof onAnalyze === 'function') {
      onAnalyze(file, options);
      return;
    }

    // 默认：本组件自己调用接口，但不渲染，只上抛结果
    setLoading(true);
    try {
      const res = await uploadAndAnalyzeImage(file, options);
      if (!res?.success) message.error(res?.message || '分析失败');
      else message.success('分析完成');
      onAnalysisFinished?.(res);
      onResults?.(res);
    } catch (e) {
      message.error(e?.message || '网络异常');
    } finally {
      setLoading(false);
    }
  };

  const previewUrl = file ? getFilePreviewUrl(file) : null;

  return (
    <Card className="upload-card" title="上传影像进行分析">
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space wrap align="center">
          <span>生成热力图：</span>
          <Switch checked={generateHeatmap} onChange={setGenerateHeatmap} />
          <span>阈值：</span>
          <InputNumber min={0} max={1} step={0.05} value={threshold} onChange={setThreshold} />
          <span>Top-K：</span>
          <InputNumber min={1} max={20} value={topK} onChange={setTopK} placeholder="留空=全部" />
          <Button type="primary" onClick={handleAnalyze} loading={loading} disabled={!file}>
            开始分析
          </Button>
        </Space>

        <Dragger
          multiple={false}
          maxCount={1}
          accept=".dcm,.jpg,.jpeg,.png"
          beforeUpload={beforeUpload}
          onChange={onChange}
          onRemove={onRemove}
          fileList={fileList}
          showUploadList={{ showPreviewIcon: false, showRemoveIcon: true }}
        >
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽影像到此处</p>
          <p className="ant-upload-hint">支持 .dcm/.jpg/.jpeg/.png，最大 50MB</p>
        </Dragger>

        {file ? (
          <Space direction="vertical" size={4}>
            <Text type="secondary">已选择：{file.name}</Text>
            {previewUrl ? <img src={previewUrl} alt="preview" style={{ maxWidth: 320, borderRadius: 8 }} /> : null}
          </Space>
        ) : null}
      </Space>
    </Card>
  );
}
