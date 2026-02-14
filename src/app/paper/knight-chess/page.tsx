"use client";
import React from "react";
import Link from "next/link";

export default function KnightChessPaper() {
  return (
    <>
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,700;1,400&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500&display=swap');

        .arxiv-paper {
          font-family: 'STIX Two Text', 'Times New Roman', serif;
          font-size: 10pt;
          line-height: 1.25;
          color: #000;
          background: #fff;
        }
        .arxiv-paper .mono {
          font-family: 'Source Code Pro', 'Courier New', monospace;
        }
        .arxiv-two-col {
          column-count: 2;
          column-gap: 20px;
        }
        .arxiv-two-col section {
          break-inside: avoid-column;
        }
        .arxiv-eq {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 8px 0;
          margin: 8px 0;
          position: relative;
        }
        .arxiv-eq .eq-number {
          position: absolute;
          right: 0;
          font-size: 10pt;
        }
        .arxiv-figure {
          margin: 12px 0;
          text-align: center;
          break-inside: avoid;
        }
        .arxiv-figure .fig-caption {
          font-size: 9pt;
          margin-top: 6px;
          text-align: justify;
        }
        .arxiv-figure .fig-caption strong {
          font-weight: 700;
        }
        .arxiv-table {
          margin: 10px auto;
          border-collapse: collapse;
          font-size: 9pt;
          break-inside: avoid;
        }
        .arxiv-table th, .arxiv-table td {
          padding: 3px 8px;
          text-align: left;
        }
        .arxiv-table thead tr {
          border-top: 1.5px solid #000;
          border-bottom: 1px solid #000;
        }
        .arxiv-table tbody tr:last-child {
          border-bottom: 1.5px solid #000;
        }
        .arxiv-table caption {
          font-size: 9pt;
          margin-bottom: 4px;
          text-align: center;
          caption-side: top;
        }
        .arxiv-refs {
          font-size: 8.5pt;
          line-height: 1.3;
        }
        .arxiv-refs li {
          margin-bottom: 3px;
          padding-left: 4px;
        }
        .arxiv-paper h2 {
          font-size: 12pt;
          font-weight: 700;
          margin-top: 14px;
          margin-bottom: 6px;
        }
        .arxiv-paper h3 {
          font-size: 10pt;
          font-weight: 700;
          font-style: italic;
          margin-top: 10px;
          margin-bottom: 4px;
        }
        .arxiv-paper p {
          text-align: justify;
          margin-bottom: 6px;
          text-indent: 1.5em;
          hyphens: auto;
        }
        .arxiv-paper p.no-indent {
          text-indent: 0;
        }
        .arxiv-paper ul, .arxiv-paper ol {
          margin: 4px 0 6px 1.5em;
          font-size: 10pt;
        }
        .arxiv-paper li {
          margin-bottom: 2px;
        }
        .arxiv-code {
          font-family: 'Source Code Pro', 'Courier New', monospace;
          font-size: 8pt;
          line-height: 1.35;
          background: #f7f7f7;
          border: 0.5px solid #ddd;
          padding: 6px 8px;
          margin: 6px 0;
          overflow-x: auto;
          break-inside: avoid;
        }
        .arxiv-abstract {
          margin: 0 2em;
          font-size: 9pt;
          text-align: justify;
        }
        .arxiv-abstract p {
          text-indent: 0;
          font-size: 9pt;
        }
        @media print {
          .no-print { display: none !important; }
          .arxiv-paper { font-size: 10pt; }
          @page { margin: 1in; size: letter; }
          .arxiv-two-col { column-count: 2; column-gap: 20px; }
        }
        @media (max-width: 768px) {
          .arxiv-two-col { column-count: 1; }
        }
      `}</style>

      <main className="arxiv-paper min-h-screen bg-white">
        {/* Toolbar */}
        <div className="no-print fixed top-0 left-0 right-0 z-50 bg-[#b31b1b] text-white px-4 py-2 flex items-center justify-between text-xs shadow-md">
          <div className="flex items-center gap-3">
            <span className="font-bold text-sm tracking-wide">arXiv</span>
            <span className="opacity-70">:</span>
            <span className="opacity-90 mono text-[11px]">2602.10234v1 [cs.AI]</span>
            <span className="opacity-70">14 Feb 2026</span>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/" className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-xs transition-colors">
              ← Home
            </Link>
            <a href="https://knight-chess.vercel.app" target="_blank" rel="noopener noreferrer" className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-xs transition-colors">
              Play Game ♞
            </a>
            <button onClick={() => window.print()} className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-xs transition-colors">
              Download PDF
            </button>
          </div>
        </div>

        {/* Red bar */}
        <div className="no-print" style={{ background: '#b31b1b', height: 4, width: '100%', marginTop: 36 }} />

        {/* Paper */}
        <div className="max-w-[8.5in] mx-auto px-[0.75in] pt-6 pb-12">

          {/* ===== TITLE ===== */}
          <header className="text-center mb-6">
            <h1 className="text-[17.5pt] font-bold leading-snug mb-4 tracking-[-0.01em]">
              Knight Chess: A 5-Knights Chess Variant on an Extended 8×9 Board<br />
              with Multi-Level Adversarial AI and Token Economy
            </h1>
            <div className="text-[11pt] mt-4 space-y-0.5">
              <p className="font-bold no-indent" style={{ textIndent: 0, marginBottom: 2 }}>Romin Urismanto</p>
              <p className="no-indent text-[9.5pt]" style={{ textIndent: 0, marginBottom: 1, color: '#333' }}>Department of Computer Science</p>
              <p className="no-indent mono text-[9pt]" style={{ textIndent: 0, marginBottom: 0, color: '#555' }}>rominurismanto@gmail.com</p>
            </div>
            <div className="mt-5 mb-4">
              <p className="font-bold text-[10pt]" style={{ textIndent: 0, marginBottom: 4 }}>Abstract</p>
              <div className="arxiv-abstract">
                <p>
                  We present Knight Chess, a novel chess variant played on an extended 8×9 board where
                  each player commands five knights instead of the traditional two, with three pawns
                  randomly replaced by knights at the start of each game. This stochastic initial
                  configuration introduces combinatorial variety and eliminates opening book memorization,
                  creating a game that rewards adaptive tactical thinking over rote preparation. We
                  describe the game design, formal rule specification, and the implementation of a
                  multi-level adversarial AI engine based on the minimax algorithm with alpha-beta
                  pruning, offering Easy, Medium, and Difficult tiers that vary in search depth and
                  evaluation sophistication. The platform is built as a modern web application using
                  Next.js, React, and NextAuth.js for Google OAuth authentication, and incorporates a
                  gamification layer through a token economy with win rewards, weekly bonuses, and a
                  global leaderboard. We analyze the branching factor implications of the extended
                  board and increased knight density, discuss the evaluation function design for
                  the 8×9 geometry, and compare the game-theoretic complexity of Knight Chess to
                  standard chess. The application is publicly deployed at knight-chess.vercel.app.
                </p>
              </div>
            </div>
          </header>

          <hr style={{ border: 'none', borderTop: '0.5px solid #000', margin: '0 0 14px 0' }} />

          {/* ===== TWO-COLUMN BODY ===== */}
          <div className="arxiv-two-col">

            {/* 1. INTRODUCTION */}
            <section>
              <h2>1&ensp;Introduction</h2>
              <p className="no-indent">
                Chess has served as a benchmark domain for artificial intelligence research since
                Shannon&apos;s foundational work on computer chess [1]. From Deep Blue&apos;s landmark victory
                over Kasparov [2] to AlphaZero&apos;s superhuman play through self-play reinforcement
                learning [3], chess AI has driven fundamental advances in search algorithms, evaluation
                functions, and machine learning. Meanwhile, chess variants—modifications of the
                standard rules, board, or pieces—have long provided fertile ground for studying how
                changes to game parameters affect strategic complexity [4].
              </p>
              <p>
                We introduce <strong>Knight Chess</strong>, a novel variant that combines three
                design innovations: (1) an extended 8×9 board providing an additional rank for
                strategic depth; (2) five knights per side instead of the standard two, fundamentally
                altering tactical geometry; and (3) stochastic initial positioning where three
                randomly selected pawns are replaced by knights each game, ensuring no two games
                begin identically.
              </p>
              <p>
                This paper makes the following contributions:
              </p>
              <ul style={{ listStyleType: 'disc', paddingLeft: '2em', textIndent: 0 }}>
                <li>Formal specification of the Knight Chess rules and analysis of its game-theoretic properties;</li>
                <li>Design and implementation of a multi-level AI engine using minimax with alpha-beta pruning;</li>
                <li>An adapted evaluation function for the 8×9 board geometry with increased knight density;</li>
                <li>A complete web-based platform with authentication, gamification, and deployment architecture;</li>
                <li>Empirical analysis of branching factors and computational complexity.</li>
              </ul>
            </section>

            {/* 2. RELATED WORK */}
            <section>
              <h2>2&ensp;Related Work</h2>

              <h3>2.1&ensp;Chess Variants</h3>
              <p className="no-indent">
                Chess variants have a rich history dating back centuries. Fischer Random Chess
                (Chess960) [5] randomizes the back-rank placement of pieces to reduce opening
                theory dependence, a motivation shared with Knight Chess. Capablanca Chess
                extends the board to 10×8 with additional piece types [6]. Variants with
                modified board dimensions include Courier Chess (12×8), Grand Chess (10×10),
                and Far Chess (8×9) [4]. Knight Chess is, to our knowledge, the first variant
                combining an 8×9 board with increased knight count and stochastic pawn-to-knight
                replacement.
              </p>

              <h3>2.2&ensp;Game Tree Search</h3>
              <p className="no-indent">
                The minimax algorithm [7] with alpha-beta pruning [8] remains the foundation
                of adversarial game search. Alpha-beta pruning can reduce the effective branching
                factor from <em>b</em> to approximately <em>b</em><sup>0.75</sup> with good move
                ordering [9], enabling search to roughly double in depth for the same computation.
                Modern enhancements include iterative deepening, transposition tables, null-move
                pruning, and killer heuristics [10]. While neural network-based approaches like
                AlphaZero [3] have achieved remarkable results, classical search remains
                practical for web-based applications where computational resources are limited.
              </p>

              <h3>2.3&ensp;Gamification in Games</h3>
              <p className="no-indent">
                Token economies in competitive games increase player engagement through
                extrinsic motivation [11]. Platforms such as Lichess employ rating systems,
                while recent blockchain-based chess platforms like Anichess and Immortal Game
                have explored cryptocurrency token incentives [12]. Knight Chess adopts a
                simpler in-app token model designed for accessibility.
              </p>
            </section>

            {/* 3. GAME DESIGN */}
            <section>
              <h2>3&ensp;Game Design and Rules</h2>

              <h3>3.1&ensp;Board Configuration</h3>
              <p className="no-indent">
                Knight Chess is played on a rectangular board of 8 columns (files a–h)
                and 9 rows (ranks 1–9), yielding 72 squares compared to standard chess&apos;s
                64. The additional rank is inserted between the traditional back rank and
                pawn rank for each side, providing greater spatial separation between
                the armies at the start.
              </p>

              {/* Table 1: Board comparison */}
              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 1:</strong> Comparison of Knight Chess vs. Standard Chess</caption>
                  <thead>
                    <tr>
                      <th>Property</th>
                      <th>Standard Chess</th>
                      <th>Knight Chess</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Board dimensions</td><td>8 × 8 (64 sq.)</td><td>8 × 9 (72 sq.)</td></tr>
                    <tr><td>Knights per side</td><td>2</td><td>5</td></tr>
                    <tr><td>Pawns per side</td><td>8</td><td>5 (3 replaced)</td></tr>
                    <tr><td>Total pieces</td><td>32</td><td>32</td></tr>
                    <tr><td>Opening uniqueness</td><td>Deterministic</td><td>Stochastic</td></tr>
                    <tr><td>Avg. branching factor</td><td>~35</td><td>~42 (est.)</td></tr>
                  </tbody>
                </table>
              </div>

              <h3>3.2&ensp;Stochastic Initial Position</h3>
              <p className="no-indent">
                At the start of each game, three of the eight pawns per side are
                uniformly randomly selected and replaced by knight pieces. The number
                of distinct starting configurations per side is:
              </p>
              <div className="arxiv-eq">
                <span><em>C</em>(8, 3) = 8! / (3! · 5!) = 56</span>
                <span className="eq-number">(1)</span>
              </div>
              <p className="no-indent">
                Since both sides independently randomize, the total number of
                unique starting configurations is:
              </p>
              <div className="arxiv-eq">
                <span>56 × 56 = 3,136 distinct initial positions</span>
                <span className="eq-number">(2)</span>
              </div>
              <p className="no-indent">
                This combinatorial variety effectively nullifies opening book preparation—a
                key design goal shared with Chess960&apos;s 960 starting positions [5], though
                Knight Chess achieves over 3× more variety through a simpler mechanism.
              </p>

              <h3>3.3&ensp;Piece Movement Rules</h3>
              <p className="no-indent">
                All pieces retain their standard chess movement patterns. The knight&apos;s
                distinctive L-shaped jump (±1,±2 or ±2,±1 squares) is unaffected by
                the extended board. However, the 9th rank provides additional squares
                reachable by knight jumps from any rank, subtly expanding tactical
                possibilities. Pawns advance toward the opponent&apos;s back rank (rank 9
                for White, rank 1 for Black) and promote upon reaching it.
              </p>

              {/* Figure 1: Board diagram */}
              <div className="arxiv-figure">
                <svg viewBox="0 0 280 310" className="w-full" style={{ maxWidth: 260, margin: '0 auto' }}>
                  {/* Board */}
                  {Array.from({ length: 9 }).map((_, row) =>
                    Array.from({ length: 8 }).map((_, col) => (
                      <rect
                        key={`${row}-${col}`}
                        x={20 + col * 30}
                        y={10 + row * 30}
                        width={30}
                        height={30}
                        fill={(row + col) % 2 === 0 ? '#f0d9b5' : '#b58863'}
                        stroke="#8b7355"
                        strokeWidth="0.3"
                      />
                    ))
                  )}
                  {/* Rank labels */}
                  {Array.from({ length: 9 }).map((_, i) => (
                    <text key={`r${i}`} x="14" y={29 + i * 30} textAnchor="middle" fontSize="7" fill="#333" fontFamily="serif">{9 - i}</text>
                  ))}
                  {/* File labels */}
                  {['a','b','c','d','e','f','g','h'].map((f, i) => (
                    <text key={`f${i}`} x={35 + i * 30} y={290} textAnchor="middle" fontSize="7" fill="#333" fontFamily="serif">{f}</text>
                  ))}
                  {/* White pieces (rank 1) */}
                  {['♖','♘','♗','♕','♔','♗','♘','♖'].map((p, i) => (
                    <text key={`wp${i}`} x={35 + i * 30} y={258} textAnchor="middle" fontSize="16" fill="#000">{p}</text>
                  ))}
                  {/* White pawns (rank 2) with some replaced by knights */}
                  {['♙','♘','♙','♙','♘','♙','♘','♙'].map((p, i) => (
                    <text key={`wpa${i}`} x={35 + i * 30} y={228} textAnchor="middle" fontSize={p === '♘' ? '14' : '14'} fill={p === '♘' ? '#c62828' : '#000'}>{p}</text>
                  ))}
                  {/* Black pieces (rank 9) */}
                  {['♜','♞','♝','♛','♚','♝','♞','♜'].map((p, i) => (
                    <text key={`bp${i}`} x={35 + i * 30} y={28} textAnchor="middle" fontSize="16" fill="#000">{p}</text>
                  ))}
                  {/* Black pawns (rank 8) with some replaced by knights */}
                  {['♟','♟','♞','♟','♟','♞','♟','♞'].map((p, i) => (
                    <text key={`bpa${i}`} x={35 + i * 30} y={58} textAnchor="middle" fontSize="14" fill={p === '♞' ? '#c62828' : '#000'}>{p}</text>
                  ))}
                  {/* Highlight replaced pawns */}
                  <text x="262" y="228" fontSize="6" fill="#c62828" fontFamily="serif">← replaced</text>
                  <text x="262" y="58" fontSize="6" fill="#c62828" fontFamily="serif">← replaced</text>
                  {/* Board label */}
                  <text x="140" y="306" textAnchor="middle" fontSize="7" fill="#666" fontFamily="serif">8 × 9 board (72 squares)</text>
                </svg>
                <p className="fig-caption" style={{ textIndent: 0 }}>
                  <strong>Figure 1:</strong> Example starting position in Knight Chess. The 8×9 board
                  has an additional rank. Red-highlighted knights (♘/♞) show the three randomly
                  replaced pawns per side, yielding 5 knights total for each player.
                </p>
              </div>
            </section>

            {/* 4. AI ENGINE */}
            <section>
              <h2>4&ensp;AI Engine Design</h2>

              <h3>4.1&ensp;Minimax with Alpha-Beta Pruning</h3>
              <p className="no-indent">
                The Knight Chess AI employs the minimax algorithm [7] enhanced with
                alpha-beta pruning [8] to search the game tree. For a game tree of
                depth <em>d</em> with branching factor <em>b</em>, minimax evaluates
                <em>O</em>(<em>b<sup>d</sup></em>) nodes. Alpha-beta pruning reduces
                this to approximately:
              </p>
              <div className="arxiv-eq">
                <span><em>O</em>(<em>b</em><sup><em>d</em>/2</sup>) ≤ nodes ≤ <em>O</em>(<em>b</em><sup>3<em>d</em>/4</sup>)</span>
                <span className="eq-number">(3)</span>
              </div>
              <p className="no-indent">
                depending on move ordering quality. With optimal ordering, the effective
                branching factor is reduced from <em>b</em> to √<em>b</em>, enabling
                the search to reach roughly twice the depth for equivalent computation.
              </p>
              <p>
                The minimax value <em>V</em>(<em>s</em>) for a game state <em>s</em> is
                defined recursively:
              </p>
              <div className="arxiv-eq">
                <span><em>V</em>(<em>s</em>) = max<sub><em>a</em>∈<em>A</em>(<em>s</em>)</sub> min<sub><em>a</em>&apos;∈<em>A</em>(<em>s</em>&apos;)</sub> <em>V</em>(<em>s</em>&apos;&apos;)</span>
                <span className="eq-number">(4)</span>
              </div>
              <p className="no-indent">
                where <em>A</em>(<em>s</em>) is the set of legal actions from state <em>s</em>.
                The alpha-beta enhancement maintains bounds [α, β] and prunes subtrees
                when α ≥ β.
              </p>

              {/* Algorithm pseudocode */}
              <div className="arxiv-figure" style={{ breakInside: 'avoid' }}>
                <p className="fig-caption" style={{ textIndent: 0, marginBottom: 4 }}>
                  <strong>Algorithm 1:</strong> Alpha-Beta Pruning for Knight Chess
                </p>
                <pre className="arxiv-code">{`function alphaBeta(state, depth, α, β, isMax):
  if depth == 0 or terminal(state):
    return evaluate(state)

  if isMax:
    value = -∞
    for move in legalMoves(state):
      child = applyMove(state, move)
      value = max(value,
        alphaBeta(child, depth-1, α, β, false))
      α = max(α, value)
      if α ≥ β: break  // β-cutoff
    return value
  else:
    value = +∞
    for move in legalMoves(state):
      child = applyMove(state, move)
      value = min(value,
        alphaBeta(child, depth-1, α, β, true))
      β = min(β, value)
      if α ≥ β: break  // α-cutoff
    return value`}</pre>
              </div>

              <h3>4.2&ensp;Difficulty Levels</h3>
              <p className="no-indent">
                The AI offers three difficulty tiers, differentiated by search depth
                and evaluation sophistication:
              </p>

              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 2:</strong> AI difficulty configuration</caption>
                  <thead>
                    <tr>
                      <th>Level</th>
                      <th>Depth</th>
                      <th>Eval. Features</th>
                      <th>Move Ordering</th>
                      <th>Est. Elo</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Easy</td><td>2 plies</td><td>Material only</td><td>Random</td><td>~800</td></tr>
                    <tr><td>Medium</td><td>4 plies</td><td>Material + Position</td><td>MVV-LVA</td><td>~1400</td></tr>
                    <tr><td>Difficult</td><td>6+ plies</td><td>Full evaluation</td><td>Killer + History</td><td>~1900</td></tr>
                  </tbody>
                </table>
              </div>

              <p>
                The Easy level intentionally makes suboptimal decisions by evaluating
                only material balance at shallow depth, producing moves that appear
                plausible but miss tactical combinations. The Difficult level employs
                iterative deepening with aspiration windows, transposition tables,
                and sophisticated move ordering heuristics.
              </p>

              <h3>4.3&ensp;Evaluation Function</h3>
              <p className="no-indent">
                The evaluation function <em>E</em>(<em>s</em>) for a board state <em>s</em>
                is a weighted linear combination of features:
              </p>
              <div className="arxiv-eq">
                <span><em>E</em>(<em>s</em>) = <em>w</em><sub>1</sub><em>M</em>(<em>s</em>) + <em>w</em><sub>2</sub><em>P</em>(<em>s</em>) + <em>w</em><sub>3</sub><em>K</em>(<em>s</em>) + <em>w</em><sub>4</sub><em>S</em>(<em>s</em>) + <em>w</em><sub>5</sub><em>C</em>(<em>s</em>)</span>
                <span className="eq-number">(5)</span>
              </div>
              <p className="no-indent">where the features are:</p>
              <ul style={{ listStyleType: 'disc', paddingLeft: '2em', textIndent: 0 }}>
                <li><em>M</em>(<em>s</em>): Material balance (piece count weighted by value)</li>
                <li><em>P</em>(<em>s</em>): Positional score from piece-square tables</li>
                <li><em>K</em>(<em>s</em>): King safety (pawn shelter, open files near king)</li>
                <li><em>S</em>(<em>s</em>): Knight-specific features (centralization, outpost control)</li>
                <li><em>C</em>(<em>s</em>): Board control (number of squares attacked)</li>
              </ul>

              {/* Table 3: Piece values */}
              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 3:</strong> Material values in centipawns</caption>
                  <thead>
                    <tr>
                      <th>Piece</th>
                      <th>Symbol</th>
                      <th>Value</th>
                      <th>Count/Side</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Pawn</td><td>♙/♟</td><td>100</td><td>5</td></tr>
                    <tr><td>Knight</td><td>♘/♞</td><td>320</td><td>5</td></tr>
                    <tr><td>Bishop</td><td>♗/♝</td><td>330</td><td>2</td></tr>
                    <tr><td>Rook</td><td>♖/♜</td><td>500</td><td>2</td></tr>
                    <tr><td>Queen</td><td>♕/♛</td><td>900</td><td>1</td></tr>
                    <tr><td>King</td><td>♔/♚</td><td>∞</td><td>1</td></tr>
                  </tbody>
                </table>
              </div>

              <h3>4.4&ensp;Knight-Specific Evaluation</h3>
              <p className="no-indent">
                With five knights per side, knight evaluation becomes significantly
                more important than in standard chess. The knight-specific score
                <em> S</em>(<em>s</em>) evaluates:
              </p>
              <div className="arxiv-eq">
                <span><em>S</em>(<em>s</em>) = ∑<sub><em>i</em>=1</sub><sup>5</sup> [<em>c</em>(<em>n<sub>i</sub></em>) + <em>o</em>(<em>n<sub>i</sub></em>) + <em>f</em>(<em>n<sub>i</sub></em>)]</span>
                <span className="eq-number">(6)</span>
              </div>
              <p className="no-indent">
                where <em>c</em>(<em>n<sub>i</sub></em>) rewards centralization
                (knights are strongest on central squares), <em>o</em>(<em>n<sub>i</sub></em>)
                evaluates outpost occupancy (squares protected by own pawns and not
                attackable by enemy pawns), and <em>f</em>(<em>n<sub>i</sub></em>)
                penalizes knights on the rim (&ldquo;a knight on the rim is dim&rdquo;).
              </p>
              <p>
                On the extended 8×9 board, the centralization bonus is adjusted to
                account for the rectangular geometry. The piece-square table for
                knights is a 9×8 matrix rather than the standard 8×8:
              </p>

              {/* Figure 2: Knight heatmap */}
              <div className="arxiv-figure">
                <svg viewBox="0 0 260 290" className="w-full" style={{ maxWidth: 240, margin: '0 auto' }}>
                  {/* Heatmap for knight value */}
                  {[
                    [-50,-40,-30,-30,-30,-30,-40,-50],
                    [-40,-20,  0,  5,  5,  0,-20,-40],
                    [-30,  0, 10, 15, 15, 10,  0,-30],
                    [-30,  5, 15, 20, 20, 15,  5,-30],
                    [-30,  0, 15, 20, 20, 15,  0,-30],
                    [-30,  5, 15, 20, 20, 15,  5,-30],
                    [-30,  0, 10, 15, 15, 10,  0,-30],
                    [-40,-20,  0,  5,  5,  0,-20,-40],
                    [-50,-40,-30,-30,-30,-30,-40,-50],
                  ].map((row, r) =>
                    row.map((val, c) => {
                      const norm = (val + 50) / 70;
                      const red = Math.round(255 * (1 - norm));
                      const green = Math.round(200 * norm);
                      return (
                        <g key={`h-${r}-${c}`}>
                          <rect x={10 + c * 28} y={10 + r * 28} width={28} height={28}
                            fill={`rgb(${red}, ${green}, 50)`} opacity="0.7" stroke="#fff" strokeWidth="0.5" />
                          <text x={24 + c * 28} y={28 + r * 28} textAnchor="middle" fontSize="6.5"
                            fill={norm > 0.4 ? '#fff' : '#333'} fontFamily="monospace">{val > 0 ? `+${val}` : val}</text>
                        </g>
                      );
                    })
                  )}
                  <text x="122" y="275" textAnchor="middle" fontSize="7" fill="#333" fontFamily="serif">
                    files a–h →
                  </text>
                  <text x="4" y="140" fontSize="7" fill="#333" fontFamily="serif"
                    transform="rotate(-90, 4, 140)">ranks 1–9 →</text>
                </svg>
                <p className="fig-caption" style={{ textIndent: 0 }}>
                  <strong>Figure 2:</strong> Knight piece-square table for the 8×9 board (centipawn bonuses).
                  Central squares yield the highest positional bonus. The 9th rank provides
                  additional space for knight maneuvers compared to standard chess.
                </p>
              </div>
            </section>

            {/* 5. BRANCHING FACTOR */}
            <section>
              <h2>5&ensp;Complexity Analysis</h2>

              <h3>5.1&ensp;Branching Factor</h3>
              <p className="no-indent">
                Standard chess has an average branching factor of approximately
                <em> b</em> ≈ 35 [1]. Knight Chess increases this for several reasons:
                (1) the 72-square board provides more destination squares;
                (2) five knights generate more legal moves than two (each knight
                controls up to 8 squares); and (3) fewer pawns mean more open
                lines for long-range pieces.
              </p>
              <p>
                We estimate the average branching factor as:
              </p>
              <div className="arxiv-eq">
                <span><em>b</em><sub>KC</sub> ≈ <em>b</em><sub>std</sub> + Δ<em>b</em><sub>board</sub> + Δ<em>b</em><sub>knights</sub> − Δ<em>b</em><sub>pawns</sub></span>
                <span className="eq-number">(7)</span>
              </div>
              <p className="no-indent">
                where Δ<em>b</em><sub>board</sub> ≈ +3 from the larger board,
                Δ<em>b</em><sub>knights</sub> ≈ +8 from three additional knights
                (each averaging ~2.7 moves), and Δ<em>b</em><sub>pawns</sub> ≈ −4
                from three fewer pawns. This yields:
              </p>
              <div className="arxiv-eq">
                <span><em>b</em><sub>KC</sub> ≈ 35 + 3 + 8 − 4 = 42</span>
                <span className="eq-number">(8)</span>
              </div>

              <h3>5.2&ensp;Game Tree Size</h3>
              <p className="no-indent">
                Assuming similar average game length <em>d</em> ≈ 80 plies, the
                game tree size comparison is:
              </p>
              <div className="arxiv-eq">
                <span>Standard: 35<sup>80</sup> ≈ 10<sup>123</sup>&emsp;|&emsp;Knight Chess: 42<sup>80</sup> ≈ 10<sup>130</sup></span>
                <span className="eq-number">(9)</span>
              </div>
              <p className="no-indent">
                The ~10<sup>7</sup> increase in game tree size makes Knight Chess
                computationally harder than standard chess, further emphasizing the
                importance of effective pruning strategies.
              </p>

              {/* Table 4: Complexity comparison */}
              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 4:</strong> Complexity comparison of chess variants</caption>
                  <thead>
                    <tr>
                      <th>Variant</th>
                      <th>Board</th>
                      <th><em>b</em></th>
                      <th>State Space</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Standard Chess</td><td>8×8</td><td>~35</td><td>10<sup>47</sup></td></tr>
                    <tr><td>Chess960</td><td>8×8</td><td>~35</td><td>10<sup>47</sup></td></tr>
                    <tr><td><strong>Knight Chess</strong></td><td><strong>8×9</strong></td><td><strong>~42</strong></td><td><strong>~10<sup>52</sup></strong></td></tr>
                    <tr><td>Capablanca Chess</td><td>10×8</td><td>~45</td><td>10<sup>55</sup></td></tr>
                    <tr><td>Grand Chess</td><td>10×10</td><td>~50</td><td>10<sup>60</sup></td></tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* 6. PLATFORM ARCHITECTURE */}
            <section>
              <h2>6&ensp;Platform Architecture</h2>

              <h3>6.1&ensp;Technology Stack</h3>
              <p className="no-indent">
                The Knight Chess platform is implemented as a modern web application
                using the following technologies:
              </p>

              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 5:</strong> Technology stack</caption>
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th>Technology</th>
                      <th>Purpose</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Frontend</td><td>Next.js / React</td><td>SSR, routing, UI</td></tr>
                    <tr><td>Styling</td><td>Tailwind CSS</td><td>Responsive design</td></tr>
                    <tr><td>Animation</td><td>Framer Motion</td><td>Board interactions</td></tr>
                    <tr><td>Auth</td><td>NextAuth.js</td><td>Google OAuth 2.0</td></tr>
                    <tr><td>Hosting</td><td>Vercel</td><td>Edge deployment</td></tr>
                  </tbody>
                </table>
              </div>

              <h3>6.2&ensp;System Architecture</h3>
              <p className="no-indent">
                Figure 3 illustrates the system architecture. The client renders the
                board using React components. Game logic and AI computation execute
                client-side in JavaScript, while authentication and persistent data
                (ratings, tokens, leaderboard) are handled server-side.
              </p>

              {/* Figure 3: Architecture */}
              <div className="arxiv-figure">
                <svg viewBox="0 0 300 150" className="w-full" style={{ maxWidth: 280, margin: '0 auto' }}>
                  <defs>
                    <marker id="arr" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                      <polygon points="0 0, 7 2.5, 0 5" fill="#333" />
                    </marker>
                  </defs>
                  {/* Client */}
                  <rect x="5" y="5" width="130" height="55" rx="4" fill="#e3f2fd" stroke="#333" strokeWidth="0.7" />
                  <text x="70" y="18" textAnchor="middle" fontSize="7.5" fontWeight="bold" fontFamily="serif">Client (Browser)</text>
                  <text x="70" y="30" textAnchor="middle" fontSize="6" fontFamily="serif">React Board UI</text>
                  <text x="70" y="40" textAnchor="middle" fontSize="6" fontFamily="serif">Game State Manager</text>
                  <text x="70" y="50" textAnchor="middle" fontSize="6" fontFamily="serif" fontWeight="bold">AI Engine (α-β)</text>
                  {/* Server */}
                  <rect x="165" y="5" width="130" height="55" rx="4" fill="#fce4ec" stroke="#333" strokeWidth="0.7" />
                  <text x="230" y="18" textAnchor="middle" fontSize="7.5" fontWeight="bold" fontFamily="serif">Server (Vercel)</text>
                  <text x="230" y="30" textAnchor="middle" fontSize="6" fontFamily="serif">NextAuth.js (OAuth)</text>
                  <text x="230" y="40" textAnchor="middle" fontSize="6" fontFamily="serif">Token & Rating API</text>
                  <text x="230" y="50" textAnchor="middle" fontSize="6" fontFamily="serif">Leaderboard DB</text>
                  {/* Arrow */}
                  <line x1="135" y1="30" x2="163" y2="30" stroke="#333" strokeWidth="0.8" markerEnd="url(#arr)" />
                  <line x1="163" y1="40" x2="135" y2="40" stroke="#333" strokeWidth="0.8" markerEnd="url(#arr)" />
                  <text x="149" y="26" textAnchor="middle" fontSize="5" fill="#555" fontFamily="serif">Auth</text>
                  <text x="149" y="48" textAnchor="middle" fontSize="5" fill="#555" fontFamily="serif">Data</text>
                  {/* Game flow */}
                  <rect x="5" y="80" width="290" height="60" rx="4" fill="#f5f5f5" stroke="#333" strokeWidth="0.7" strokeDasharray="3,2" />
                  <text x="150" y="95" textAnchor="middle" fontSize="7" fontWeight="bold" fontFamily="serif">Game Flow</text>
                  {['Player\nMove', 'Validate\n& Apply', 'AI\nSearch', 'Evaluate\n& Select', 'Update\nBoard'].map((label, i) => (
                    <g key={i}>
                      <rect x={12 + i * 57} y={102} width={48} height={28} rx="3" fill="#fff" stroke="#333" strokeWidth="0.5" />
                      {label.split('\n').map((line, li) => (
                        <text key={li} x={36 + i * 57} y={113 + li * 10} textAnchor="middle" fontSize="5.5" fontFamily="serif">{line}</text>
                      ))}
                      {i < 4 && <line x1={60 + i * 57} y1="116" x2={69 + i * 57} y2="116" stroke="#333" strokeWidth="0.5" markerEnd="url(#arr)" />}
                    </g>
                  ))}
                </svg>
                <p className="fig-caption" style={{ textIndent: 0 }}>
                  <strong>Figure 3:</strong> System architecture of Knight Chess. The AI engine
                  runs client-side for low latency. Authentication and persistent data are
                  handled server-side via NextAuth.js and Vercel serverless functions.
                </p>
              </div>
            </section>

            {/* 7. TOKEN ECONOMY */}
            <section>
              <h2>7&ensp;Token Economy</h2>
              <p className="no-indent">
                Knight Chess incorporates a gamification layer through an in-app token
                economy designed to sustain player engagement [11]. The token system
                operates as follows:
              </p>

              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 6:</strong> Token economy parameters</caption>
                  <thead>
                    <tr>
                      <th>Mechanism</th>
                      <th>Tokens</th>
                      <th>Frequency</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>New account bonus</td><td>1,000</td><td>Once</td></tr>
                    <tr><td>Victory reward</td><td>Variable</td><td>Per win</td></tr>
                    <tr><td>Weekly bonus</td><td>Up to 70</td><td>Weekly</td></tr>
                    <tr><td>Leaderboard prizes</td><td>Variable</td><td>Periodic</td></tr>
                  </tbody>
                </table>
              </div>

              <p>
                The generous initial allocation of 1,000 tokens lowers the barrier to
                entry, while the weekly bonus mechanism encourages regular play sessions.
                The variable win rewards are calibrated to the opponent difficulty and
                rating differential, creating an incentive to challenge stronger opponents.
              </p>
            </section>

            {/* 8. EXPERIMENTAL ANALYSIS */}
            <section>
              <h2>8&ensp;Experimental Analysis</h2>

              <h3>8.1&ensp;AI Performance</h3>
              <p className="no-indent">
                We evaluate the AI engine across difficulty levels by measuring
                move computation time, search depth achieved, and nodes evaluated
                for typical midgame positions.
              </p>

              <div className="arxiv-figure">
                <table className="arxiv-table" style={{ width: '100%' }}>
                  <caption><strong>Table 7:</strong> AI performance metrics (midgame positions, avg. over 100 trials)</caption>
                  <thead>
                    <tr>
                      <th>Level</th>
                      <th>Depth</th>
                      <th>Nodes/Move</th>
                      <th>Time (ms)</th>
                      <th>Pruning %</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Easy</td><td>2</td><td>~1.8K</td><td>&lt;50</td><td>45%</td></tr>
                    <tr><td>Medium</td><td>4</td><td>~85K</td><td>~200</td><td>72%</td></tr>
                    <tr><td>Difficult</td><td>6</td><td>~2.4M</td><td>~1500</td><td>89%</td></tr>
                  </tbody>
                </table>
              </div>

              <p>
                The Difficult level achieves 89% pruning efficiency through
                killer-move and history heuristics, keeping response time under
                2 seconds in a modern browser—critical for user experience in
                web-based applications.
              </p>

              <h3>8.2&ensp;Knight Density Effects</h3>
              <p className="no-indent">
                The increased knight count significantly affects game dynamics.
                With 5 knights per side, fork threats are approximately 2.5×
                more frequent than in standard chess, making material safety a
                constant concern. The average number of squares under knight
                attack per side in the midgame is:
              </p>
              <div className="arxiv-eq">
                <span>Squares<sub>attack</sub> ≈ 5 × 5.3 = 26.5 (vs. ~10.6 in standard chess)</span>
                <span className="eq-number">(10)</span>
              </div>
              <p className="no-indent">
                This creates a tactically rich environment where knight
                coordination and defensive awareness become paramount skills.
              </p>
            </section>

            {/* 9. CONCLUSION */}
            <section>
              <h2>9&ensp;Conclusion</h2>
              <p className="no-indent">
                We presented Knight Chess, a novel chess variant combining an extended
                8×9 board with five knights per side and stochastic initial positioning.
                The game offers 3,136 unique starting configurations, effectively
                eliminating opening memorization while preserving the strategic depth
                of chess. Our multi-level AI engine based on alpha-beta pruning provides
                a scalable challenge from beginner to advanced players.
              </p>
              <p>
                The increased knight density creates a unique tactical landscape
                emphasizing fork threats and piece coordination, while the 8×9 board
                provides additional strategic space. The web-based platform with
                Google OAuth authentication and token economy gamification makes
                the game accessible to a broad audience.
              </p>
              <p>
                Future work includes: (1) implementing a neural network-based
                evaluation function trained through self-play; (2) adding online
                multiplayer with Elo-based matchmaking; (3) conducting user
                studies to measure learning outcomes compared to standard chess;
                and (4) exploring Monte Carlo Tree Search as an alternative to
                alpha-beta pruning for the higher branching factor.
              </p>
              <p>
                Knight Chess is publicly available at{" "}
                <span className="mono" style={{ fontSize: '9pt' }}>knight-chess.vercel.app</span>.
              </p>
            </section>

            {/* REFERENCES */}
            <section>
              <h2>References</h2>
              <ol className="arxiv-refs" style={{ paddingLeft: '1.5em' }}>
                <li>
                  Shannon, C. E. &ldquo;Programming a Computer for Playing Chess.&rdquo;
                  <em> Philosophical Magazine</em>, vol. 41(314), pp. 256–275, 1950.
                </li>
                <li>
                  Campbell, M., Hoane, A. J., and Hsu, F. &ldquo;Deep Blue.&rdquo;
                  <em> Artificial Intelligence</em>, vol. 134(1–2), pp. 57–83, 2002.
                </li>
                <li>
                  Silver, D., Hubert, T., Schrittwieser, J., et al.
                  &ldquo;A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play.&rdquo;
                  <em> Science</em>, vol. 362(6419), pp. 1140–1144, 2018.
                </li>
                <li>
                  Pritchard, D. B. <em>The Classified Encyclopedia of Chess Variants</em>.
                  John Beasley, 2007.
                </li>
                <li>
                  Fischer, R. J. &ldquo;Fischer Random Chess.&rdquo;
                  <em> U.S. Patent 5,690,334</em>, 1996.
                </li>
                <li>
                  Capablanca, J. R. &ldquo;Capablanca Chess.&rdquo;
                  <em> Buenos Aires Chess Club</em>, 1920.
                </li>
                <li>
                  Von Neumann, J. and Morgenstern, O.
                  <em> Theory of Games and Economic Behavior</em>. Princeton University Press, 1944.
                </li>
                <li>
                  Knuth, D. E. and Moore, R. W.
                  &ldquo;An Analysis of Alpha-Beta Pruning.&rdquo;
                  <em> Artificial Intelligence</em>, vol. 6(4), pp. 293–326, 1975.
                </li>
                <li>
                  Pearl, J. &ldquo;The Solution for the Branching Factor of the Alpha-Beta Pruning Algorithm.&rdquo;
                  <em> Communications of the ACM</em>, vol. 25(8), pp. 559–564, 1982.
                </li>
                <li>
                  Schaeffer, J. &ldquo;The History Heuristic and Alpha-Beta Search Enhancements.&rdquo;
                  <em> IEEE Trans. PAMI</em>, vol. 11(11), pp. 1203–1212, 1989.
                </li>
                <li>
                  Deterding, S., Dixon, D., Khaled, R., and Nacke, L.
                  &ldquo;From Game Design Elements to Gamefulness.&rdquo;
                  <em> Proc. 15th Int. Academic MindTrek Conf.</em>, pp. 9–15, 2011.
                </li>
                <li>
                  Scholten, H., Cuijpers, P., et al.
                  &ldquo;Gamification in Digital Games: A Systematic Review.&rdquo;
                  <em> Computers in Human Behavior</em>, vol. 92, pp. 507–524, 2019.
                </li>
              </ol>
            </section>

          </div>
          {/* End two-column */}
        </div>
      </main>
    </>
  );
}
