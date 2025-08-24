import React, { useState, useEffect } from 'react';
import { Layout, Menu, Card, Row, Col, Statistic, Progress, Table, Space, Tag, Button, Spin } from 'antd';
import { 
  DashboardOutlined, 
  UserOutlined, 
  ShoppingCartOutlined, 
  BarChartOutlined,
  RobotOutlined,
  HeartOutlined,
  ThunderboltOutlined,
  EyeOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';
import './App.css';

const { Header, Sider, Content } = Layout;

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Mock data for demo purposes
const mockRealtimeData = {
  activeUsers: 1247,
  totalRevenue: 89432.50,
  conversionRate: 3.7,
  avgOrderValue: 127.65,
  agentPerformance: {
    customer: { efficiency: 92, responseTime: 45 },
    product: { efficiency: 88, responseTime: 52 },
    recommendation: { efficiency: 95, responseTime: 38 }
  }
};

const mockChartData = [
  { time: '00:00', users: 120, revenue: 2400, recommendations: 45 },
  { time: '04:00', users: 89, revenue: 1890, recommendations: 32 },
  { time: '08:00', users: 340, revenue: 6800, recommendations: 98 },
  { time: '12:00', users: 567, revenue: 11340, recommendations: 156 },
  { time: '16:00', users: 789, revenue: 15780, recommendations: 234 },
  { time: '20:00', users: 432, revenue: 8640, recommendations: 123 }
];

const mockTopProducts = [
  { key: '1', name: 'Smart Laptop Pro', views: 1234, conversions: 45, revenue: 44550 },
  { key: '2', name: 'Wireless Headphones', views: 987, conversions: 78, revenue: 15522 },
  { key: '3', name: 'Gaming Mouse Elite', views: 765, conversions: 34, revenue: 2720 },
  { key: '4', name: 'Ultrawide Monitor', views: 543, conversions: 12, revenue: 7176 },
  { key: '5', name: 'Mechanical Keyboard', views: 432, conversions: 23, revenue: 3220 }
];

const customerSegmentData = [
  { name: 'Tech Enthusiasts', value: 35, color: '#8884d8' },
  { name: 'Price Sensitive', value: 25, color: '#82ca9d' },
  { name: 'Luxury Shoppers', value: 20, color: '#ffc658' },
  { name: 'Frequent Buyers', value: 15, color: '#ff7300' },
  { name: 'New Visitors', value: 5, color: '#00ff00' }
];

function SmartShoppingDashboard() {
  const [loading, setLoading] = useState(true);
  const [selectedMenu, setSelectedMenu] = useState('overview');
  const [realtimeData, setRealtimeData] = useState(mockRealtimeData);
  const [systemStatus, setSystemStatus] = useState('operational');

  useEffect(() => {
    // Simulate API calls and real-time updates
    const fetchData = async () => {
      try {
        setLoading(true);
        // In production, replace with actual API calls
        // const response = await axios.get(`${API_BASE_URL}/api/analytics/overview`);
        
        // Simulate loading delay
        setTimeout(() => {
          setRealtimeData(mockRealtimeData);
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error('Failed to fetch data:', error);
        setLoading(false);
      }
    };

    fetchData();

    // Set up real-time updates
    const interval = setInterval(() => {
      setRealtimeData(prev => ({
        ...prev,
        activeUsers: prev.activeUsers + Math.floor(Math.random() * 20 - 10),
        totalRevenue: prev.totalRevenue + Math.random() * 1000,
        conversionRate: Math.max(0, prev.conversionRate + (Math.random() * 0.4 - 0.2))
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const renderOverview = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Users"
              value={realtimeData.activeUsers}
              prefix={<UserOutlined style={{ color: '#1890ff' }} />}
              suffix={<Tag color="green">+12%</Tag>}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Revenue"
              value={realtimeData.totalRevenue}
              prefix="$"
              precision={2}
              suffix={<Tag color="blue">+28%</Tag>}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Conversion Rate"
              value={realtimeData.conversionRate}
              suffix="%"
              prefix={<ThunderboltOutlined style={{ color: '#52c41a' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Avg Order Value"
              value={realtimeData.avgOrderValue}
              prefix="$"
              precision={2}
              suffix={<Tag color="orange">+15%</Tag>}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card title="Real-time Performance Metrics" extra={<Tag color="green">Live</Tag>}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="users" stroke="#8884d8" strokeWidth={2} />
                <Line type="monotone" dataKey="recommendations" stroke="#82ca9d" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="Customer Segments">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={customerSegmentData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {customerSegmentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderAgentMonitoring = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card title="Customer Agent" extra={<Tag color="green">Active</Tag>}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div style={{ marginBottom: 8 }}>Efficiency: {realtimeData.agentPerformance.customer.efficiency}%</div>
                <Progress percent={realtimeData.agentPerformance.customer.efficiency} status="active" />
              </div>
              <div>
                <div style={{ marginBottom: 8 }}>Response Time: {realtimeData.agentPerformance.customer.responseTime}ms</div>
                <Progress 
                  percent={100 - realtimeData.agentPerformance.customer.responseTime} 
                  status="active" 
                  strokeColor="#52c41a"
                />
              </div>
            </Space>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="Product Agent" extra={<Tag color="green">Active</Tag>}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div style={{ marginBottom: 8 }}>Efficiency: {realtimeData.agentPerformance.product.efficiency}%</div>
                <Progress percent={realtimeData.agentPerformance.product.efficiency} status="active" />
              </div>
              <div>
                <div style={{ marginBottom: 8 }}>Response Time: {realtimeData.agentPerformance.product.responseTime}ms</div>
                <Progress 
                  percent={100 - realtimeData.agentPerformance.product.responseTime} 
                  status="active" 
                  strokeColor="#52c41a"
                />
              </div>
            </Space>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="Recommendation Agent" extra={<Tag color="green">Active</Tag>}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div style={{ marginBottom: 8 }}>Efficiency: {realtimeData.agentPerformance.recommendation.efficiency}%</div>
                <Progress percent={realtimeData.agentPerformance.recommendation.efficiency} status="active" />
              </div>
              <div>
                <div style={{ marginBottom: 8 }}>Response Time: {realtimeData.agentPerformance.recommendation.responseTime}ms</div>
                <Progress 
                  percent={100 - realtimeData.agentPerformance.recommendation.responseTime} 
                  status="active" 
                  strokeColor="#52c41a"
                />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="Agent Coordination Timeline">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={mockChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="recommendations" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );

  const renderProductAnalytics = () => (
    <div>
      <Card title="Top Performing Products" extra={<Button type="primary">Export Data</Button>}>
        <Table
          columns={[
            { title: 'Product Name', dataIndex: 'name', key: 'name' },
            { title: 'Views', dataIndex: 'views', key: 'views', sorter: true },
            { title: 'Conversions', dataIndex: 'conversions', key: 'conversions', sorter: true },
            { 
              title: 'Conversion Rate', 
              key: 'conversionRate',
              render: (_, record) => `${((record.conversions / record.views) * 100).toFixed(2)}%`
            },
            { 
              title: 'Revenue', 
              dataIndex: 'revenue', 
              key: 'revenue', 
              render: value => `$${value.toLocaleString()}`,
              sorter: true
            }
          ]}
          dataSource={mockTopProducts}
          pagination={false}
        />
      </Card>
    </div>
  );

  const menuItems = [
    { key: 'overview', icon: <DashboardOutlined />, label: 'Overview' },
    { key: 'agents', icon: <RobotOutlined />, label: 'Agent Monitoring' },
    { key: 'products', icon: <ShoppingCartOutlined />, label: 'Product Analytics' },
    { key: 'customers', icon: <UserOutlined />, label: 'Customer Insights' },
    { key: 'recommendations', icon: <HeartOutlined />, label: 'Recommendations' }
  ];

  const renderContent = () => {
    if (loading) {
      return (
        <div style={{ textAlign: 'center', padding: '100px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>Loading dashboard data...</div>
        </div>
      );
    }

    switch (selectedMenu) {
      case 'overview': return renderOverview();
      case 'agents': return renderAgentMonitoring();
      case 'products': return renderProductAnalytics();
      default: return renderOverview();
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={200} theme="dark">
        <div style={{ padding: '16px', color: 'white', textAlign: 'center', borderBottom: '1px solid #001529' }}>
          <RobotOutlined style={{ fontSize: '24px', marginRight: 8 }} />
          Smart Shopping AI
        </div>
        <Menu
          mode="inline"
          theme="dark"
          selectedKeys={[selectedMenu]}
          items={menuItems}
          onClick={({ key }) => setSelectedMenu(key)}
          style={{ height: '100%', borderRight: 0 }}
        />
      </Sider>
      
      <Layout>
        <Header style={{ background: '#fff', padding: '0 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0 }}>Multi-Agent AI Dashboard</h2>
          <Space>
            <Tag color={systemStatus === 'operational' ? 'green' : 'red'}>
              System {systemStatus}
            </Tag>
            <Button type="primary" icon={<EyeOutlined />}>
              Live Demo
            </Button>
          </Space>
        </Header>
        
        <Content style={{ margin: '24px', background: '#f0f2f5' }}>
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  );
}

export default SmartShoppingDashboard;