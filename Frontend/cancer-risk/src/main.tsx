import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import react from 'react'
import Button from './Button.jsx'


createRoot(document.getElementById('root')!).render(
  <StrictMode>

    <App />
    <Button />

  </StrictMode>
)