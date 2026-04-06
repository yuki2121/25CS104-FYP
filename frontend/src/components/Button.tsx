import React from 'react';

const Button = ({ label, onClick, className, type = 'button' }) => {
  return (
    <button
      type={type} 
      onClick={onClick}
      className = {`rounded bg-blue-500 text-white hover:bg-purple-600 m-2 focus:outline-none focus:ring-2 ${className}`}
    >
      {label}
    </button>
  );
};

export default Button;