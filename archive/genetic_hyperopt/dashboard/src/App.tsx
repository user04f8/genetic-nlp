import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid,
} from 'recharts';

interface ModelStatus {
  model_id: string;
  epoch: number;
  val_acc: number;
  hyperparameters: { [key: string]: any };
}

function App() {
  const [data, setData] = useState<{ [key: string]: ModelStatus[] }>({});

  useEffect(() => {
    const fetchData = async () => {
      const result = await axios('/status');
      const statuses: ModelStatus[] = result.data;

      // Organize data by model_id
      const modelData: { [key: string]: ModelStatus[] } = {};
      statuses.forEach((status) => {
        const { model_id } = status;
        if (!modelData[model_id]) {
          modelData[model_id] = [];
        }
        modelData[model_id].push(status);
      });

      // Sort epochs for each model
      Object.keys(modelData).forEach((model_id) => {
        modelData[model_id].sort((a, b) => a.epoch - b.epoch);
      });

      setData(modelData);
    };
    fetchData();
  }, []);

  const colors = ['#8884d8', '#82ca9d', '#ff7300', '#8884d8', '#82ca9d', '#ff7300'];

  return (
    <div>
      <h1>Model Validation Accuracy Over Epochs</h1>
      <LineChart width={1000} height={600}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -10 }} />
        <YAxis domain={[0, 1]} label={{ value: 'Validation Accuracy', angle: -90, position: 'insideLeft' }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />

        {Object.keys(data).map((model_id, index) => (
          <Line
            key={model_id}
            data={data[model_id]}
            dataKey="val_acc"
            name={model_id}
            stroke={colors[index % colors.length]}
            dot={false}
          />
        ))}
      </LineChart>
    </div>
  );
}

// Custom Tooltip to display hyperparameters
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const status: ModelStatus = payload[0].payload;
    const hyperparameters = status.hyperparameters;
    return (
      <div style={{ backgroundColor: 'white', border: '1px solid #ccc', padding: '10px' }}>
        <p><strong>Model ID:</strong> {status.model_id}</p>
        <p><strong>Epoch:</strong> {status.epoch}</p>
        <p><strong>Validation Accuracy:</strong> {status.val_acc.toFixed(4)}</p>
        <p><strong>Hyperparameters:</strong></p>
        <ul>
          {Object.keys(hyperparameters).map((key) => (
            <li key={key}><strong>{key}:</strong> {hyperparameters[key]}</li>
          ))}
        </ul>
      </div>
    );
  }

  return null;
};

export default App;
