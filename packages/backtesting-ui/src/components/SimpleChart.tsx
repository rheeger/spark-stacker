import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

// Sample data for demonstration
const data = [
  { name: 'Jan', value: 400, profit: 240 },
  { name: 'Feb', value: 300, profit: 138 },
  { name: 'Mar', value: 600, profit: 380 },
  { name: 'Apr', value: 800, profit: 580 },
  { name: 'May', value: 500, profit: 250 },
  { name: 'Jun', value: 900, profit: 650 },
  { name: 'Jul', value: 700, profit: 450 },
];

interface SimpleChartProps {
  title?: string;
}

export const SimpleChart: React.FC<SimpleChartProps> = ({
  title = 'Sample Chart',
}) => {
  return (
    <div className="w-full h-full">
      <h2 className="text-xl font-bold mb-4 text-gray-800">{title}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            name="Trading Volume"
            stroke="#8884d8"
            activeDot={{ r: 8 }}
          />
          <Line
            type="monotone"
            dataKey="profit"
            name="Profit"
            stroke="#82ca9d"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SimpleChart;
