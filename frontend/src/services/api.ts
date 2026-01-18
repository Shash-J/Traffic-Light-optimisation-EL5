/**
 * API Service
 * HTTP client for REST API calls
 */
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface SimulationRequest {
    mode: 'fixed' | 'rl';
    use_gui: boolean;
    traffic_scenario?: 'peak' | 'offpeak';
    location?: string;
}

export interface SimulationResponse {
    status: string;
    message: string;
    mode?: string;
    traffic_scenario?: string;
    location?: string;
}

export interface LocationConfig {
    name: string;
    description: string;
    real_world_stats: {
        avg_wait_time: number;
        avg_queue_length: number;
        vehicle_flow_per_hour: number;
    };
    simulation_config: any;
}

export type LocationsMap = Record<string, LocationConfig>;

export interface SimulationStatus {
    running: boolean;
    pid: number | null;
    traci_connected: boolean;
    active_connections: number;
    current_metrics: any;
}

export interface MetricsSummary {
    status: string;
    summary?: {
        total_vehicles: number;
        average_queue: number;
        average_waiting_time: number;
        simulation_time: number;
        throughput: {
            departed: number;
            arrived: number;
        };
    };
    message?: string;
}

/**
 * Get available locations
 */
export const getLocations = async (): Promise<LocationsMap> => {
    const response = await apiClient.get<LocationsMap>('/api/simulation/locations');
    return response.data;
};

/**
 * Start simulation
 */
export const startSimulation = async (
    mode: 'fixed' | 'rl',
    useGui: boolean = true,
    trafficScenario: 'peak' | 'offpeak' = 'peak',
    location: string = 'silk_board'
): Promise<SimulationResponse> => {
    const response = await apiClient.post<SimulationResponse>('/api/simulation/start', {
        mode,
        use_gui: useGui,
        traffic_scenario: trafficScenario,
        location
    });
    return response.data;
};

/**
 * Stop simulation
 */
export const stopSimulation = async (): Promise<SimulationResponse> => {
    const response = await apiClient.post<SimulationResponse>('/api/simulation/stop');
    return response.data;
};

/**
 * Reset simulation
 */
export const resetSimulation = async (): Promise<SimulationResponse> => {
    const response = await apiClient.post<SimulationResponse>('/api/simulation/reset');
    return response.data;
};

/**
 * Get simulation status
 */
export const getSimulationStatus = async (): Promise<SimulationStatus> => {
    const response = await apiClient.get<SimulationStatus>('/api/simulation/status');
    return response.data;
};

/**
 * Get current metrics
 */
export const getCurrentMetrics = async () => {
    const response = await apiClient.get('/api/metrics/current');
    return response.data;
};

/**
 * Get metrics summary
 */
export const getMetricsSummary = async (): Promise<MetricsSummary> => {
    const response = await apiClient.get<MetricsSummary>('/api/metrics/summary');
    return response.data;
};

// ============== ADVANCED FEATURES API ==============

export interface WeatherState {
    condition: number;
    condition_name: string;
    speed_factor: number;
    min_green_adjustment: number;
    is_raining: boolean;
}

export interface EmergencyVehicle {
    id: string;
    type: 'AMBULANCE' | 'FIRE_TRUCK' | 'POLICE';
    distance: number;
    eta: number;
    junction: string;
    priority: number;
}

export interface EmergencyStatus {
    active: boolean;
    vehicle: EmergencyVehicle | null;
    preemption_active: boolean;
    time_remaining: number;
}

export interface ControllerMetrics {
    controller: string;
    avg_delay: number;
    throughput: number;
    emergency_time: number;
    queue_length: number;
}

export interface EvaluationMetrics {
    comparison: ControllerMetrics[];
    improvement: {
        delay_reduction: number;
        throughput_gain: number;
    };
    total_steps: number;
}

/**
 * Get current weather conditions
 */
export const getWeatherStatus = async (): Promise<WeatherState> => {
    const response = await apiClient.get<WeatherState>('/api/advanced/weather');
    return response.data;
};

/**
 * Set weather condition (for testing)
 */
export const setWeatherCondition = async (condition: number): Promise<void> => {
    await apiClient.post('/api/advanced/weather/set', null, {
        params: { condition }
    });
};

/**
 * Get emergency vehicle status
 */
export const getEmergencyStatus = async (): Promise<EmergencyStatus> => {
    const response = await apiClient.get<EmergencyStatus>('/api/advanced/emergency');
    return response.data;
};

/**
 * Simulate emergency vehicle (for testing)
 */
export const simulateEmergency = async (
    vehicleType: 'AMBULANCE' | 'FIRE_TRUCK' | 'POLICE' = 'AMBULANCE',
    distance: number = 200,
    junction: string = 'silk_board'
): Promise<void> => {
    await apiClient.post('/api/advanced/emergency/simulate', null, {
        params: {
            vehicle_type: vehicleType,
            distance,
            junction
        }
    });
};

/**
 * Clear emergency status
 */
export const clearEmergency = async (): Promise<void> => {
    await apiClient.post('/api/advanced/emergency/clear');
};

/**
 * Get evaluation/comparison metrics
 */
export const getEvaluationMetrics = async (): Promise<EvaluationMetrics> => {
    const response = await apiClient.get<EvaluationMetrics>('/api/advanced/evaluation/comparison');
    return response.data;
};

export default apiClient;
