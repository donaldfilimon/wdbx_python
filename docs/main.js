// Main Application Component
const App = () => {
  // Hooks and State Management
  const [state, setState] = React.useState({
    darkMode: localStorage.getItem("darkMode") === "true" || false,
    expandedCategories: JSON.parse(
      localStorage.getItem("expandedCategories"),
    ) || {
      "core-classes": true,
      "storage-classes": true,
      "concurrency-classes": true,
      "ai-components": true,
      "database-clients": true,
      "utility-classes": true,
    },
    showBackToTop: false,
    searchTerm: "",
    showMobileMenu: false,
    copySuccess: null,
    activeTab: "api",
    showShortcutsModal: false,
    lastVisitedSections:
      JSON.parse(localStorage.getItem("lastVisitedSections")) || [],
    fontSize: localStorage.getItem("fontSize") || "medium",
    bookmarks: JSON.parse(localStorage.getItem("bookmarks")) || [],
    searchResults: [],
    showToast: false,
    toastMessage: "",
  });

  // Actions and Operations
  const actions = React.useMemo(
    () => ({
      toggleDarkMode: () => {
        setState((prev) => {
          const newDarkMode = !prev.darkMode;
          document.body.classList[newDarkMode ? "add" : "remove"]("dark-mode");
          localStorage.setItem("darkMode", newDarkMode);
          return { ...prev, darkMode: newDarkMode };
        });
      },

      toggleCategory: (category) => {
        setState((prev) => {
          const newCategories = {
            ...prev.expandedCategories,
            [category]: !prev.expandedCategories[category],
          };
          localStorage.setItem(
            "expandedCategories",
            JSON.stringify(newCategories),
          );
          return { ...prev, expandedCategories: newCategories };
        });
      },

      toggleAllCategories: (expand) => {
        const allCategories = {};
        categories.forEach((category) => (allCategories[category.id] = expand));
        localStorage.setItem(
          "expandedCategories",
          JSON.stringify(allCategories),
        );
        setState((prev) => ({ ...prev, expandedCategories: allCategories }));
      },

      updateSearchTerm: (term) => {
        setState((prev) => ({ ...prev, searchTerm: term }));
      },

      setActiveTab: (tab) => {
        setState((prev) => ({ ...prev, activeTab: tab }));
      },

      toggleMobileMenu: () => {
        setState((prev) => ({ ...prev, showMobileMenu: !prev.showMobileMenu }));
      },

      changeFontSize: (size) => {
        document.documentElement.setAttribute("data-font-size", size);
        localStorage.setItem("fontSize", size);
        setState((prev) => ({ ...prev, fontSize: size }));
        actions.showToast(`Font size changed to ${size}`);
      },

      toggleBookmark: (id, title) => {
        setState((prev) => {
          const exists = prev.bookmarks.some((b) => b.id === id);
          const newBookmarks = exists
            ? prev.bookmarks.filter((b) => b.id !== id)
            : [...prev.bookmarks, { id, title, timestamp: Date.now() }];

          localStorage.setItem("bookmarks", JSON.stringify(newBookmarks));
          actions.showToast(
            `${exists ? "Removed" : "Added"} "${title}" ${exists ? "from" : "to"} bookmarks`,
          );
          return { ...prev, bookmarks: newBookmarks };
        });
      },

      showToast: (message, type = "info") => {
        setState((prev) => ({
          ...prev,
          showToast: true,
          toastMessage: { text: message, type },
        }));
        setTimeout(() => {
          setState((prev) => ({ ...prev, showToast: false }));
        }, 3000);
      },

      scrollToTop: () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      },
    }),
    [],
  );

  // Side Effects
  React.useEffect(() => {
    // System preference detection
    if (state.darkMode) {
      document.body.classList.add("dark-mode");
    }
    document.documentElement.setAttribute("data-font-size", state.fontSize);

    const prefersDarkMode = window.matchMedia(
      "(prefers-color-scheme: dark)",
    ).matches;
    if (prefersDarkMode && localStorage.getItem("darkMode") === null) {
      actions.toggleDarkMode();
    }
  }, [state.darkMode, state.fontSize, actions]);

  // Scroll handling
  React.useEffect(() => {
    const handleScroll = () => {
      setState((prev) => ({ ...prev, showBackToTop: window.scrollY > 300 }));

      const sections = document.querySelectorAll("section[id]");
      const scrollPosition = window.scrollY + 200;

      sections.forEach((section) => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute("id");

        if (
          scrollPosition >= sectionTop &&
          scrollPosition <= sectionTop + sectionHeight
        ) {
          setState((prev) => {
            const filtered = prev.lastVisitedSections.filter(
              (item) => item !== sectionId,
            );
            const newList = [sectionId, ...filtered].slice(0, 5);
            localStorage.setItem(
              "lastVisitedSections",
              JSON.stringify(newList),
            );
            return { ...prev, lastVisitedSections: newList };
          });
        }
      });
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Search functionality
  React.useEffect(() => {
    const searchTimeout = setTimeout(() => {
      if (state.searchTerm.trim()) {
        const results = [];
        let foundInContent = false;

        // Search implementation...
        categories.forEach((category) => {
          const categoryMatch = category.title
            .toLowerCase()
            .includes(state.searchTerm.toLowerCase());
          category.items.forEach((item) => {
            if (
              item.toLowerCase().includes(state.searchTerm.toLowerCase()) ||
              categoryMatch
            ) {
              results.push({ category: category.id, item });
            }
          });
        });

        document
          .querySelectorAll(".card-body p, .method-body p, h3, h4")
          .forEach((el) => {
            if (
              el.textContent
                .toLowerCase()
                .includes(state.searchTerm.toLowerCase())
            ) {
              foundInContent = true;
              let parent = el.closest("[id]");
              if (parent) {
                const id = parent.getAttribute("id");
                const category = categories.find((cat) =>
                  cat.items.some((item) => item.toLowerCase() === id),
                );
                if (category) {
                  results.push({
                    category: category.id,
                    item: id.charAt(0).toUpperCase() + id.slice(1),
                  });
                }
              }
            }
          });

        const uniqueResults = Array.from(
          new Set(results.map((r) => `${r.category}-${r.item}`)),
        ).map((key) => {
          const [category, item] = key.split("-");
          return { category, item };
        });

        setState((prev) => {
          const newState = { ...prev, searchResults: uniqueResults };
          if (uniqueResults.length > 0) {
            const newCategories = { ...prev.expandedCategories };
            uniqueResults.forEach((result) => {
              newCategories[result.category] = true;
            });
            return { ...newState, expandedCategories: newCategories };
          }
          return newState;
        });

        if (foundInContent && uniqueResults.length === 0) {
          actions.showToast(
            "Found matches in content. Use browser search (Ctrl+F) to locate.",
          );
        }
      } else {
        setState((prev) => ({ ...prev, searchResults: [] }));
      }
    }, 300);

    return () => clearTimeout(searchTimeout);
  }, [state.searchTerm, categories]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        document
          .querySelector('input[aria-label="Search documentation"]')
          .focus();
      }
      if (
        e.key === "Escape" &&
        document.activeElement.getAttribute("aria-label") ===
          "Search documentation"
      ) {
        actions.updateSearchTerm("");
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "/") {
        e.preventDefault();
        setState((prev) => ({ ...prev, showShortcutsModal: true }));
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "b") {
        e.preventDefault();
        actions.toggleDarkMode();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "ArrowUp") {
        e.preventDefault();
        actions.scrollToTop();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [actions]);

  // Memoized Components
  const ShortcutsModal = React.useMemo(
    () => (
      <div
        className={`modal fade ${state.showShortcutsModal ? "show d-block" : ""}`}
        tabIndex="-1"
      >
        <div className="modal-dialog modal-dialog-centered">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Keyboard Shortcuts</h5>
              <button
                type="button"
                className="btn-close"
                onClick={() =>
                  setState((prev) => ({ ...prev, showShortcutsModal: false }))
                }
              ></button>
            </div>
            <div className="modal-body">
              <table className="table table-striped">
                <thead>
                  <tr>
                    <th>Shortcut</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <kbd>Ctrl</kbd> + <kbd>K</kbd>
                    </td>
                    <td>Focus search</td>
                  </tr>
                  <tr>
                    <td>
                      <kbd>Esc</kbd>
                    </td>
                    <td>Clear search (while focused)</td>
                  </tr>
                  <tr>
                    <td>
                      <kbd>Ctrl</kbd> + <kbd>/</kbd>
                    </td>
                    <td>Show this help</td>
                  </tr>
                  <tr>
                    <td>
                      <kbd>Ctrl</kbd> + <kbd>B</kbd>
                    </td>
                    <td>Toggle dark/light mode</td>
                  </tr>
                  <tr>
                    <td>
                      <kbd>Ctrl</kbd> + <kbd>↑</kbd>
                    </td>
                    <td>Scroll to top</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() =>
                  setState((prev) => ({ ...prev, showShortcutsModal: false }))
                }
              >
                Close
              </button>
            </div>
          </div>
        </div>
        <div className="modal-backdrop fade show"></div>
      </div>
    ),
    [state.showShortcutsModal],
  );

  const Toast = React.useMemo(
    () => (
      <div
        className={`toast position-fixed ${state.showToast ? "show" : ""}`}
        style={{ bottom: "20px", left: "20px", zIndex: 1060 }}
      >
        <div
          className={`toast-header ${state.toastMessage.type === "error" ? "bg-danger text-white" : "bg-primary text-white"}`}
        >
          <i
            className={`bi bi-${state.toastMessage.type === "error" ? "exclamation-circle" : "info-circle"} me-2`}
          ></i>
          <strong className="me-auto">
            {state.toastMessage.type === "error" ? "Error" : "Info"}
          </strong>
          <button
            type="button"
            className="btn-close btn-close-white"
            onClick={() => setState((prev) => ({ ...prev, showToast: false }))}
          ></button>
        </div>
        <div className="toast-body">{state.toastMessage.text}</div>
      </div>
    ),
    [state.showToast, state.toastMessage],
  );

  const Sidebar = React.useMemo(
    () => (
      <aside
        className={`sidebar ${state.showMobileMenu ? "mobile-visible" : ""}`}
      >
        <h1 className="mb-3">WDBX API Documentation</h1>
        <div className="sidebar-version text-muted mb-4">Version 1.0.0</div>

        <div className="input-group mb-3 position-relative">
          <span className="input-group-text bg-light">
            <i className="bi bi-search"></i>
          </span>
          <input
            type="text"
            className="form-control"
            placeholder="Search... (Ctrl+K)"
            value={state.searchTerm}
            onChange={(e) => actions.updateSearchTerm(e.target.value)}
            aria-label="Search documentation"
          />
          {state.searchTerm && (
            <button
              className="btn btn-sm btn-outline-secondary position-absolute"
              style={{
                right: "5px",
                top: "50%",
                transform: "translateY(-50%)",
                zIndex: 5,
              }}
              onClick={() => actions.updateSearchTerm("")}
              aria-label="Clear search"
            >
              <i className="bi bi-x"></i>
            </button>
          )}
        </div>

        {state.searchResults.length > 0 && (
          <div className="alert alert-info mb-3">
            <small>
              Found {state.searchResults.length} result
              {state.searchResults.length !== 1 ? "s" : ""}
            </small>
          </div>
        )}

        <div className="mb-3">
          <div className="btn-group btn-group-sm w-100">
            <button
              className="btn btn-outline-secondary"
              onClick={() => actions.toggleAllCategories(true)}
            >
              <i className="bi bi-chevron-down me-1"></i> Expand All
            </button>
            <button
              className="btn btn-outline-secondary"
              onClick={() => actions.toggleAllCategories(false)}
            >
              <i className="bi bi-chevron-up me-1"></i> Collapse All
            </button>
          </div>
        </div>

        {state.bookmarks.length > 0 && (
          <div className="card mb-3">
            <div className="card-header bg-primary text-white">
              <i className="bi bi-bookmark-star me-2"></i> Bookmarks
            </div>
            <div className="list-group list-group-flush">
              {state.bookmarks.map((bookmark) => (
                <a
                  key={bookmark.id}
                  href={`#${bookmark.id}`}
                  className="list-group-item list-group-item-action d-flex justify-content-between align-items-center"
                >
                  <span>{bookmark.title}</span>
                  <button
                    className="btn btn-sm btn-outline-danger"
                    onClick={(e) => {
                      e.preventDefault();
                      actions.toggleBookmark(bookmark.id, bookmark.title);
                    }}
                  >
                    <i className="bi bi-x"></i>
                  </button>
                </a>
              ))}
            </div>
          </div>
        )}

        {state.lastVisitedSections.length > 0 && (
          <div className="card mb-3">
            <div className="card-header bg-info text-white">
              <i className="bi bi-clock-history me-2"></i> Recently Viewed
            </div>
            <div className="list-group list-group-flush">
              {state.lastVisitedSections.map((sectionId) => {
                const matchedCategory = categories.find(
                  (cat) =>
                    cat.id === sectionId ||
                    cat.items.some((item) => item.toLowerCase() === sectionId),
                );
                const title = matchedCategory
                  ? matchedCategory.id === sectionId
                    ? matchedCategory.title
                    : matchedCategory.items.find(
                        (item) => item.toLowerCase() === sectionId,
                      )
                  : sectionId.charAt(0).toUpperCase() + sectionId.slice(1);

                return (
                  <a
                    key={sectionId}
                    href={`#${sectionId}`}
                    className="list-group-item list-group-item-action py-2"
                  >
                    {title}
                  </a>
                );
              })}
            </div>
          </div>
        )}

        <div className="list-group">
          {categories.map((category) => (
            <React.Fragment key={category.id}>
              <button
                className="list-group-item list-group-item-action d-flex justify-content-between align-items-center"
                onClick={() => actions.toggleCategory(category.id)}
                role="button"
                aria-expanded={state.expandedCategories[category.id]}
                tabIndex={0}
                onKeyDown={(e) =>
                  e.key === "Enter" && actions.toggleCategory(category.id)
                }
              >
                <div>
                  {category.title}
                  {category.description && (
                    <small className="d-block text-muted">
                      {category.description}
                    </small>
                  )}
                </div>
                <i
                  className={`bi bi-chevron-right ${state.expandedCategories[category.id] ? "expanded" : ""}`}
                ></i>
              </button>

              {state.expandedCategories[category.id] && (
                <div className="list-group ms-3">
                  {category.items
                    .filter(
                      (item) =>
                        !state.searchTerm ||
                        item
                          .toLowerCase()
                          .includes(state.searchTerm.toLowerCase()),
                    )
                    .map((item) => (
                      <a
                        key={item}
                        href={`#${item.toLowerCase()}`}
                        className="list-group-item list-group-item-action py-2 d-flex justify-content-between align-items-center"
                        onClick={() => actions.toggleMobileMenu(false)}
                      >
                        <span>{item}</span>
                        <div>
                          {state.searchTerm &&
                            item
                              .toLowerCase()
                              .includes(state.searchTerm.toLowerCase()) && (
                              <span className="badge bg-primary rounded-pill me-2">
                                match
                              </span>
                            )}
                          <button
                            className="btn btn-sm btn-link p-0"
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              actions.toggleBookmark(item.toLowerCase(), item);
                            }}
                            title={
                              state.bookmarks.some(
                                (b) => b.id === item.toLowerCase(),
                              )
                                ? "Remove bookmark"
                                : "Add bookmark"
                            }
                          >
                            <i
                              className={`bi bi-bookmark${state.bookmarks.some((b) => b.id === item.toLowerCase()) ? "-fill text-primary" : ""}`}
                            ></i>
                          </button>
                        </div>
                      </a>
                    ))}
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </aside>
    ),
    [
      state.showMobileMenu,
      state.searchTerm,
      state.searchResults,
      state.expandedCategories,
      state.bookmarks,
      state.lastVisitedSections,
      actions,
      categories,
    ],
  );

  // Main render
  return (
    <div className="container-fluid p-0">
      <div className="row g-0">
        <button
          className="btn btn-primary position-fixed d-md-none"
          style={{ top: "10px", left: "10px", zIndex: 1030 }}
          onClick={() => actions.toggleMobileMenu()}
          aria-label={state.showMobileMenu ? "Close menu" : "Open menu"}
          aria-expanded={state.showMobileMenu}
        >
          <i className={`bi bi-${state.showMobileMenu ? "x" : "list"}`}></i>{" "}
          Menu
        </button>

        <div
          className={`col-md-3 col-lg-2 d-md-block ${state.showMobileMenu ? "d-block" : "d-none"}`}
          style={{ height: "100vh", overflowY: "auto" }}
        >
          {Sidebar}
        </div>

        <main className="col-md-9 col-lg-10 px-md-4 py-4">
          <div className="d-flex justify-content-between mb-4 flex-wrap">
            <div className="btn-group mb-2">
              <button
                className={`btn ${state.activeTab === "api" ? "btn-primary" : "btn-outline-primary"}`}
                onClick={() => actions.setActiveTab("api")}
                aria-pressed={state.activeTab === "api"}
              >
                <i className="bi bi-code-square me-2"></i>API Reference
              </button>
              <button
                className={`btn ${state.activeTab === "guides" ? "btn-primary" : "btn-outline-primary"}`}
                onClick={() => actions.setActiveTab("guides")}
                aria-pressed={state.activeTab === "guides"}
              >
                <i className="bi bi-book me-2"></i>Guides
              </button>
              <button
                className={`btn ${state.activeTab === "examples" ? "btn-primary" : "btn-outline-primary"}`}
                onClick={() => actions.setActiveTab("examples")}
                aria-pressed={state.activeTab === "examples"}
              >
                <i className="bi bi-file-code me-2"></i>Examples
              </button>
            </div>

            <div className="dropdown mb-2 me-2">
              <button
                className="btn btn-outline-secondary dropdown-toggle"
                type="button"
                id="accessibilityDropdown"
                data-bs-toggle="dropdown"
              >
                <i className="bi bi-text-size me-2"></i>Font Size
              </button>
              <ul
                className="dropdown-menu"
                aria-labelledby="accessibilityDropdown"
              >
                <li>
                  <button
                    className="dropdown-item"
                    onClick={() => actions.changeFontSize("small")}
                  >
                    Small
                  </button>
                </li>
                <li>
                  <button
                    className="dropdown-item"
                    onClick={() => actions.changeFontSize("medium")}
                  >
                    Medium
                  </button>
                </li>
                <li>
                  <button
                    className="dropdown-item"
                    onClick={() => actions.changeFontSize("large")}
                  >
                    Large
                  </button>
                </li>
              </ul>
            </div>

            <button
              className="btn btn-outline-secondary mb-2"
              onClick={actions.toggleDarkMode}
              aria-label={
                state.darkMode ? "Switch to light mode" : "Switch to dark mode"
              }
            >
              <i
                className={`bi bi-${state.darkMode ? "sun" : "moon"} me-2`}
              ></i>
              {state.darkMode ? "Light Mode" : "Dark Mode"}
            </button>
          </div>

          <h1 className="display-5 fw-bold mb-3">WDBX API Documentation</h1>
          <p className="lead mb-5">
            This document provides a comprehensive guide to the WDBX API,
            including detailed descriptions of all classes, methods, and their
            parameters.
          </p>

          {/* Documentation sections with individual class cards */}
          {/* (Content omitted for brevity - includes detailed documentation of all classes) */}
        </main>
      </div>

      {ShortcutsModal}
      {Toast}

      {state.showBackToTop && (
        <button
          className="btn btn-primary rounded-circle position-fixed"
          style={{
            bottom: "20px",
            right: "20px",
            width: "40px",
            height: "40px",
          }}
          onClick={actions.scrollToTop}
          aria-label="Back to top"
        >
          <i className="bi bi-arrow-up"></i>
        </button>
      )}
    </div>
  );
};
// Render the app
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
