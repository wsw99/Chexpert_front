// src/features/history/components/HistoryTable.jsx
import React, { useEffect, useState } from 'react';
import { Table, Tag, Progress } from 'antd';
import { fetchAnalysisHistory } from '../api';

export default function HistoryTable() {
  const [data, setData] = useState([]);
  const [total, setTotal] = useState(0);
  const [pagination, setPagination] = useState({ current: 1, pageSize: 10 });
  const [loading, setLoading] = useState(false);

  const load = async (page = 1, pageSize = 10) => {
    setLoading(true);
    try {
      const res = await fetchAnalysisHistory({ page, pageSize });
      setData(res.items.map((it, idx) => ({ key: `${page}-${idx}`, ...it })));
      setTotal(res.total);
      setPagination({ current: page, pageSize });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(pagination.current, pagination.pageSize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const columns = [
    {
      title: 'Date',
      dataIndex: 'date',
      sorter: (a, b) => a.date.localeCompare(b.date),
      defaultSortOrder: 'descend',
      width: 140,
    },
    {
      title: 'File Name',
      dataIndex: 'fileName',
      ellipsis: true,
    },
    {
      title: 'Diagnosis',
      dataIndex: 'diagnosis',
      width: 200,
      render: (v) => {
        const norm = (v || '').toLowerCase();
        const color = norm === 'no finding' || norm === 'normal' ? 'green' : 'orange';
        return <Tag color={color} style={{ fontWeight: 600 }}>{v || '-'}</Tag>;
      },
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      width: 180,
      render: (v) => <Progress percent={Math.round((v || 0) * 100)} size="small" />,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      width: 140,
      render: (v) => <Tag color="success">{v || 'completed'}</Tag>,
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={data}
      loading={loading}
      pagination={{
        current: pagination.current,
        pageSize: pagination.pageSize,
        total,
        showSizeChanger: true,
        pageSizeOptions: ['5', '10', '20', '50'],
        onChange: (page, pageSize) => load(page, pageSize),
      }}
      rowKey="key"
    />
  );
}
