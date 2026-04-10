/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Source Sans 3'", "system-ui", "sans-serif"],
        display: ["'Fraunces'", "Georgia", "serif"],
      },
      colors: {
        ink: { 950: "#0c1222", 900: "#121a2e", 800: "#1a2540" },
        accent: { DEFAULT: "#2563eb", muted: "#93c5fd" },
      },
    },
  },
  plugins: [],
};
