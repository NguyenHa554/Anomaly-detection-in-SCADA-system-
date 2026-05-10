import { createContext, useContext } from 'react';

export const ScadaStreamContext = createContext(null);

export function useScadaStream() {
    const value = useContext(ScadaStreamContext);
    if (!value) {
        throw new Error('useScadaStream must be used inside ScadaStreamProvider');
    }
    return value;
}
