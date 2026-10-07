#!/usr/bin/env bash
# Install Neat GenAI Studio from its own repository.
#
#   curl -fsSL https://raw.githubusercontent.com/dotimothy/neat-genai-studio/main/install.sh | bash
#
# This fetches the source into ./neat-genai-studio and stops there; run
# ./setup.sh in that directory next (or pass STUDIO_SETUP=1 to do it here).
#
# Environment:
#   STUDIO_DIR      Where to install. Default: ./neat-genai-studio
#                   (also the first argument: ... | bash -s -- /path/to/dir)
#   STUDIO_BRANCH   Branch or tag to install. Default: main
#   STUDIO_REPO     GitHub repository as owner/name.
#                   Default: dotimothy/neat-genai-studio
#   GITHUB_TOKEN    Token for a private repository.
#   STUDIO_ARCHIVE  A .tar.gz of the repository to install from instead of
#                   downloading (a local path or a URL), e.g. for an offline board.
#   STUDIO_SETUP=1  Run ./setup.sh once the source is in place.
#
# Everything lives in main() so a download cut short cannot run half a script.
set -euo pipefail

main() {
  local repo="${STUDIO_REPO:-dotimothy/neat-genai-studio}"
  local branch="${STUDIO_BRANCH:-main}"
  local dir="${1:-${STUDIO_DIR:-neat-genai-studio}}"
  local archive="${STUDIO_ARCHIVE:-}"

  say()  { printf '%s\n' "$*"; }
  fail() { printf 'install: %s\n' "$*" >&2; exit 1; }

  command -v tar >/dev/null 2>&1 || fail "tar is required."

  if [[ -e "${dir}" ]]; then
    if [[ -f "${dir}/run.sh" ]]; then
      fail "${dir} already holds an install. Update it with: cd ${dir} && ./run.sh update"
    fi
    if [[ ! -d "${dir}" || -n "$(ls -A "${dir}" 2>/dev/null)" ]]; then
      fail "${dir} exists and is not empty. Choose another place with STUDIO_DIR=/path."
    fi
  fi

  local tmp
  tmp="$(mktemp -d)" || fail "mktemp failed."
  # shellcheck disable=SC2064  # expand now: tmp is local to main
  trap "rm -rf '${tmp}'" EXIT

  if [[ -n "${archive}" && -f "${archive}" ]]; then
    say "Installing Neat GenAI Studio from ${archive}"
    cp "${archive}" "${tmp}/src.tar.gz" || fail "cannot read ${archive}"
  else
    local url="${archive}"
    local -a auth=()
    if [[ -z "${url}" ]]; then
      if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        # A private repository: only the API endpoint accepts a token.
        url="https://api.github.com/repos/${repo}/tarball/${branch}"
      else
        url="https://github.com/${repo}/archive/${branch}.tar.gz"
      fi
    fi
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=( -H "Authorization: Bearer ${GITHUB_TOKEN}" )
    say "Fetching Neat GenAI Studio (${repo}, ${branch})"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL ${auth[@]+"${auth[@]}"} "${url}" -o "${tmp}/src.tar.gz" \
        || fail "download failed: ${url}$([[ -z "${GITHUB_TOKEN:-}" ]] && printf ' (a private repository needs GITHUB_TOKEN)')"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "${tmp}/src.tar.gz" ${GITHUB_TOKEN:+--header="Authorization: Bearer ${GITHUB_TOKEN}"} "${url}" \
        || fail "download failed: ${url}"
    else
      fail "curl or wget is required."
    fi
  fi

  # The archive holds one top-level directory with the Studio in it. List it to
  # a file first: under pipefail, `tar | grep -q` fails when grep stops reading
  # at its first match and tar is cut off.
  tar -tzf "${tmp}/src.tar.gz" > "${tmp}/list" 2>/dev/null || fail "not a readable .tar.gz archive."
  if ! grep -qE '^[^/]+/run\.sh$' "${tmp}/list"; then
    fail "that archive is not Neat GenAI Studio (no run.sh at its top level)."
  fi
  mkdir -p "${dir}"
  tar -xzf "${tmp}/src.tar.gz" -C "${dir}" --strip-components=1 || fail "extract failed."
  chmod +x "${dir}/run.sh" "${dir}/setup.sh" 2>/dev/null || true

  say "Installed in ${dir}"
  if [[ "${STUDIO_SETUP:-0}" == "1" ]]; then
    say "Running setup…"
    ( cd "${dir}" && ./setup.sh )
  else
    say ""
    say "Next:"
    say "  cd ${dir}"
    say "  ./setup.sh     # environments, speech model, voices, local config"
    say "  ./run.sh       # start the Studio"
  fi
}

main "$@"
