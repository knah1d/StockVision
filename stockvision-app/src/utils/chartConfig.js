// Chart.js global configuration
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register all Chart.js components globally
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

// Chart cleanup utility
export const destroyChart = (chartRef) => {
  if (chartRef && chartRef.current) {
    try {
      chartRef.current.destroy();
    } catch (error) {
      console.warn('Error destroying chart:', error);
    }
  }
};

// Default chart options
export const defaultChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
    },
    tooltip: {
      enabled: true,
    },
  },
  interaction: {
    intersect: false,
    mode: 'index',
  },
};

export default ChartJS;
