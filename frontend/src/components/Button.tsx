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
    <button type={type} onClick={onClick} className={className} style={style}>
      {label}
    </button>
  );
}