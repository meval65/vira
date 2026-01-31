import type { ReactNode } from 'react';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'info';
}

export function Toast({ message, type = 'info' }: ToastProps): ReactNode {
  return <div className={`toast ${type}`} role="alert">{message}</div>;
}
