import { useEffect, useRef, useState } from "react";

const portraitSrc = "/_assets/media/portrait.webp";
const lightSrc = "/_assets/media/hero-light-720.webp";

function ContactIcon({ type }) {
  const paths = {
    mail: (
      <>
        <path d="M4.75 6.75h14.5v10.5H4.75z" />
        <path d="m5.25 7.25 6.75 5.5 6.75-5.5" />
      </>
    ),
    phone: (
      <path d="M8.25 5.25 10 9l-1.5 1.15a9.2 9.2 0 0 0 4.35 4.35L14 13l3.75 1.75-.9 3a1.25 1.25 0 0 1-1.35.88C9.85 17.92 6.08 14.15 5.37 8.5a1.25 1.25 0 0 1 .88-1.35z" />
    ),
    linkedin: (
      <>
        <path d="M6.5 9.5v7" />
        <path d="M10.25 16.5v-3.75a3.25 3.25 0 0 1 6.5 0v3.75" />
        <path d="M10.25 9.5v7" />
        <path d="M6.5 6.25v.05" />
      </>
    ),
  };

  return (
    <svg className="contact-icon" viewBox="0 0 24 24" aria-hidden="true">
      {paths[type]}
    </svg>
  );
}

export default function App() {
  const [isContactOpen, setIsContactOpen] = useState(false);
  const scrollAnimationRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle("contact-open", isContactOpen);
    document.body.classList.toggle("contact-open", isContactOpen);

    if (isContactOpen) {
      requestAnimationFrame(() => {
        const scrollRoot = document.getElementById("app-root");
        if (!scrollRoot) {
          return;
        }

        if (scrollAnimationRef.current) {
          cancelAnimationFrame(scrollAnimationRef.current);
        }

        const start = scrollRoot.scrollTop;
        const end = scrollRoot.scrollHeight - scrollRoot.clientHeight;
        const distance = end - start;
        const duration = 920;
        const startedAt = performance.now();

        const easeOutCubic = (progress) => 1 - Math.pow(1 - progress, 3);

        const animate = (now) => {
          const progress = Math.min((now - startedAt) / duration, 1);
          scrollRoot.scrollTop = start + distance * easeOutCubic(progress);

          if (progress < 1) {
            scrollAnimationRef.current = requestAnimationFrame(animate);
          }
        };

        scrollAnimationRef.current = requestAnimationFrame(animate);
      });
    }

    return () => {
      if (scrollAnimationRef.current) {
        cancelAnimationFrame(scrollAnimationRef.current);
      }

      document.documentElement.classList.remove("contact-open");
      document.body.classList.remove("contact-open");
    };
  }, [isContactOpen]);

  return (
    <main className={`site${isContactOpen ? " is-contact-open" : ""}`}>
      <section className="hero" aria-labelledby="hero-title">
        <img
          className="hero-light hero-light-top"
          src={lightSrc}
          alt=""
          aria-hidden="true"
          width="720"
          height="681"
          decoding="async"
          fetchPriority="low"
        />

        <div className="hero-copy">
          <p className="eyebrow">Vinayak Verma</p>
          <h1 id="hero-title">
            Building the bridge from AI models to <em>silicon</em>.
          </h1>
          <p className="intro">
            I'm building NexSilica to turn trained AI models into optimized custom silicon. My work
            spans semiconductor research, chip design, and software automation.
          </p>
        </div>

        <div className="hero-art" aria-hidden="true">
          <img
            className="portrait"
            src={portraitSrc}
            alt=""
            width="1065"
            height="1477"
            decoding="async"
            fetchPriority="high"
          />
        </div>

        <button
          className="contact-link"
          type="button"
          aria-expanded={isContactOpen}
          aria-controls="contact"
          onClick={() => setIsContactOpen((isOpen) => !isOpen)}
        >
          <span className="contact-rule" />
          <span className="contact-label">Contact Me</span>
          <span className="contact-rule" />
        </button>
      </section>
      <section id="contact" className="contact-section" aria-label="Contact details">
        <div className="contact-panel">
          <div className="contact-heading">
            <p className="contact-kicker">Reach me directly</p>
            <p className="contact-note">Email, call, or connect on LinkedIn.</p>
          </div>
          <div className="contact-actions">
            <a href="mailto:vinayakverma@nexsilica.com">
              <ContactIcon type="mail" />
              <span>
                <small>Email</small>
                vinayakverma@nexsilica.com
              </span>
            </a>
            <a href="tel:+918527090612">
              <ContactIcon type="phone" />
              <span>
                <small>Phone</small>
                +91 8527090612
              </span>
            </a>
            <a href="https://www.linkedin.com/in/vinayakverma/">
              <ContactIcon type="linkedin" />
              <span>
                <small>LinkedIn</small>
                vinayakverma
              </span>
            </a>
          </div>
        </div>
      </section>
    </main>
  );
}
