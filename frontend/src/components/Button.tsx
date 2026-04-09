import type { CSSProperties } from "react";

type ButtonProps = {
  label: string;
  onClick: () => void;
  className?: string;
  type?: "button" | "submit" | "reset";
  style?: CSSProperties;
};

export default function Button({
  label,
  onClick,
  className = "",
  type = "button",
  style,
}: ButtonProps) {
  return (
    <button type={type} onClick={onClick} className = {`rounded bg-blue-500 text-white hover:bg-purple-600 m-2 focus:outline-none focus:ring-2 ${className}`} style={style}>
      {label}
    </button>
  );
}