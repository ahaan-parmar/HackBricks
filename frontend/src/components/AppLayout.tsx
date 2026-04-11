import { Link, useLocation } from "react-router-dom";

const navLinks = [
  { to: "/", label: "Dashboard" },
  { to: "/students", label: "Students" },
  { to: "/segments", label: "Segments" },
  { to: "/fairness", label: "Fairness" },
  { to: "/reports", label: "Reports" },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-nav text-nav-foreground h-12 flex items-center px-6">
        <Link to="/" className="font-bold text-heading4 mr-12 tracking-tight">
          Drop(In)
        </Link>
        <nav className="flex gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`px-3 py-1.5 text-body font-medium transition-colors ${
                location.pathname === link.to
                  ? "text-nav-foreground"
                  : "text-nav-foreground/60 hover:text-nav-foreground/80"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </header>
      <main className="px-6 py-6 max-w-[1400px] mx-auto">{children}</main>
    </div>
  );
}
