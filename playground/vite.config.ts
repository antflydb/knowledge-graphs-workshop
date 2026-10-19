// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Antfly's native inference server (`make inference`).
const antfly = process.env.ANTFLY_URL ?? "http://127.0.0.1:8090";

export default defineConfig({
  plugins: [react()],
  // Served by the workshop web app at /playground/, which forwards /ai to inference.
  base: "/playground/",
  build: { outDir: "../kg/web/static/playground", emptyOutDir: true },
  server: {
    port: 5317,
    proxy: { "/ai": { target: antfly, changeOrigin: true } },
  },
});
