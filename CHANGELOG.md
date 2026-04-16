# Changelog

## [0.15.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.14.0...letta-evals-v0.15.0) (2026-04-16)


### Features

* Add compute_gate_score() for per-sample reward computation ([#243](https://github.com/letta-ai/letta-evals/issues/243)) ([65ab71f](https://github.com/letta-ai/letta-evals/commit/65ab71fd2d653abf461ee2ea99788fde5ce05c2b))

## [0.14.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.13.0...letta-evals-v0.14.0) (2026-04-14)


### Features

* Training-loop improvements for letta-code target ([#237](https://github.com/letta-ai/letta-evals/issues/237)) ([e6f6946](https://github.com/letta-ai/letta-evals/commit/e6f69462abc61b815670e987fcef11d8c7fc922c))
* **visualization:** Add live target cost to rich progress ([#236](https://github.com/letta-ai/letta-evals/issues/236)) ([d7740dc](https://github.com/letta-ai/letta-evals/commit/d7740dcb8af543f32c1b92b64a3a97609421e5f2))


### Bug Fixes

* Allow for longer inputs to letta code target by piping from stdin ([#222](https://github.com/letta-ai/letta-evals/issues/222)) ([3179b1f](https://github.com/letta-ai/letta-evals/commit/3179b1fc0d2a622bdfe4ca64dc516888fa1fb245))
* Don't use --yolo when permission_mode is set ([#238](https://github.com/letta-ai/letta-evals/issues/238)) ([57491f7](https://github.com/letta-ai/letta-evals/commit/57491f72f34e8339e9a218545bc9695b3b408988))
* Remove duplicate error logs and use consistent 0-indexed sample IDs ([#242](https://github.com/letta-ai/letta-evals/issues/242)) ([857a993](https://github.com/letta-ai/letta-evals/commit/857a993e23690ab90ddf1ffc990678e4d6ae3d42))
* Set decorator metadata attributes on wrapper instead of original function ([#241](https://github.com/letta-ai/letta-evals/issues/241)) ([dac8faa](https://github.com/letta-ai/letta-evals/commit/dac8faa7f2797a74e79c239beba4e01ab8571383))

## [0.13.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.12.0...letta-evals-v0.13.0) (2026-04-03)


### Features

* Improve Rich progress stability under concurrency ([#224](https://github.com/letta-ai/letta-evals/issues/224)) ([c816825](https://github.com/letta-ai/letta-evals/commit/c8168258de94cb4a0f93a3b062ee5ee9eb7407c0))


### Bug Fixes

* Clean up imported letta judge agents ([#235](https://github.com/letta-ai/letta-evals/issues/235)) ([f90bfe5](https://github.com/letta-ai/letta-evals/commit/f90bfe555345dedc199d1ac87fdfd5c58d30d11a))
* **examples:** Use OpenAI for simple rubric grader ([#234](https://github.com/letta-ai/letta-evals/issues/234)) ([54e0ff2](https://github.com/letta-ai/letta-evals/commit/54e0ff207032496c58eef6adf49683ac72c621c0))
* **visualization:** Improve rich progress visualization layout ([#233](https://github.com/letta-ai/letta-evals/issues/233)) ([9ac3bef](https://github.com/letta-ai/letta-evals/commit/9ac3bef7b3e3b82d3a0f4d045b7ce8231b0fb6e4))


### Refactors

* Cleanup dead Rich progress paths ([#226](https://github.com/letta-ai/letta-evals/issues/226)) ([fcb5571](https://github.com/letta-ai/letta-evals/commit/fcb55711fcdb60fdefaeca306d3e43e9e191d900))
* Extract rich progress renderer ([#229](https://github.com/letta-ai/letta-evals/issues/229)) ([433a150](https://github.com/letta-ai/letta-evals/commit/433a1500f5e6b17114285c5f9be7d7fa26ceddc8))
* Extract rich progress state reducer ([#228](https://github.com/letta-ai/letta-evals/issues/228)) ([5f7a1c6](https://github.com/letta-ai/letta-evals/commit/5f7a1c6ebf0c7aebd77f9452fcd37499b35d4f86))
* extract visualization summary helpers ([#227](https://github.com/letta-ai/letta-evals/issues/227)) ([7d0dbe1](https://github.com/letta-ai/letta-evals/commit/7d0dbe1503911b08617e628a79ca7457f96d0e55))
* Remove dead visualization code ([#231](https://github.com/letta-ai/letta-evals/issues/231)) ([d82d0dc](https://github.com/letta-ai/letta-evals/commit/d82d0dc4308bfeeab09886a68fa4aa2c483c1f3e))
* Remove stale rich progress api ([#230](https://github.com/letta-ai/letta-evals/issues/230)) ([a624c70](https://github.com/letta-ai/letta-evals/commit/a624c70d11e0ba22f043a8ffeb0770a38ef756dd))

## [0.12.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.11.0...letta-evals-v0.12.0) (2026-04-01)


### Features

* Add cleanup attribute to SuiteSpec for post-eval agent deletion ([#223](https://github.com/letta-ai/letta-evals/issues/223)) ([eb3d8e9](https://github.com/letta-ai/letta-evals/commit/eb3d8e9df4dded135ba684cfb146faf17193b9e2))
* Add per-sample and aggregate time tracking to evals ([#201](https://github.com/letta-ai/letta-evals/issues/201)) ([1b78755](https://github.com/letta-ai/letta-evals/commit/1b787553811f8be5d108e898feeb94676e2795d4))
* Filesystem v2 leaderboard refresh ([#217](https://github.com/letta-ai/letta-evals/issues/217)) ([ace75e3](https://github.com/letta-ai/letta-evals/commit/ace75e3d2fe993b790eb8f42a56f10cdc462e827))
* Write run logs to output directory for post-run debugging ([#221](https://github.com/letta-ai/letta-evals/issues/221)) ([f728067](https://github.com/letta-ai/letta-evals/commit/f728067b59761014d953d25381a7313008cd343f))


### Bug Fixes

* Fix incorrect and ambiguous filesystem agent samples ([#216](https://github.com/letta-ai/letta-evals/issues/216)) ([e614026](https://github.com/letta-ai/letta-evals/commit/e6140262abf970b7ea94bcac548b52a9dd7a7e2b))
* Include last stdout event in LettaCodeTarget error on rc!=0 ([#220](https://github.com/letta-ai/letta-evals/issues/220)) ([e9d57a9](https://github.com/letta-ai/letta-evals/commit/e9d57a90d6f6f2ccb49fa28c4f097a98c5ce3a52))
* Normalize retry behavior in LettaCodeTarget ([#210](https://github.com/letta-ai/letta-evals/issues/210)) ([09a0367](https://github.com/letta-ai/letta-evals/commit/09a0367ef72294d512dd7e26fda8e6e5201d027e))


### Refactors

* DRY up _calculate_metrics with aggregation helpers ([#204](https://github.com/letta-ai/letta-evals/issues/204)) ([4e2fecd](https://github.com/letta-ai/letta-evals/commit/4e2fecd06023915e9236d406631acb26eb8d493d))
* Extract _detect_errors from run_sample ([#211](https://github.com/letta-ai/letta-evals/issues/211)) ([7829970](https://github.com/letta-ai/letta-evals/commit/7829970fe960cc3c69c382ed6401b65cb6a483a3))
* Extract _extract_model_name helper to DRY up 3 blocks ([#209](https://github.com/letta-ai/letta-evals/issues/209)) ([a99b63c](https://github.com/letta-ai/letta-evals/commit/a99b63c60ce42cc40c50562d06460c15288cbe6a))
* Extract _parse_json_dict_field to DRY up CSV loader ([#212](https://github.com/letta-ai/letta-evals/issues/212)) ([e77c5c8](https://github.com/letta-ai/letta-evals/commit/e77c5c84c593b35b8a1d217e021febcef134ff81))
* Extract grader boilerplate into base class ([#203](https://github.com/letta-ai/letta-evals/issues/203)) ([2ad19b7](https://github.com/letta-ai/letta-evals/commit/2ad19b75b6eff26792a3f8ff01099559bc81984a))
* Extract grading logic from run_sample into helper methods ([#206](https://github.com/letta-ai/letta-evals/issues/206)) ([e5b564e](https://github.com/letta-ai/letta-evals/commit/e5b564e927c7e88d424dd3ca8923fa8e6be67f7e))
* Extract metrics computation into separate module ([#205](https://github.com/letta-ai/letta-evals/issues/205)) ([7788c40](https://github.com/letta-ai/letta-evals/commit/7788c4022419222ef99b719496782ead70eeb20e))
* Replace error categorization heuristic with phase tracking ([#208](https://github.com/letta-ai/letta-evals/issues/208)) ([341b636](https://github.com/letta-ai/letta-evals/commit/341b636d90b28a336ce2c3516edb066b8cac29ae))

## [0.11.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.10.0...letta-evals-v0.11.0) (2026-02-26)


### Features

* Add analyzing-eval-errors skill ([#199](https://github.com/letta-ai/letta-evals/issues/199)) ([f3a9ce6](https://github.com/letta-ai/letta-evals/commit/f3a9ce66ffeb56b750ee3fe5c3e19e320803d89a))
* Add flags attribute to LettaCodeTargetSpec ([#186](https://github.com/letta-ai/letta-evals/issues/186)) ([0b63014](https://github.com/letta-ai/letta-evals/commit/0b630142c9cc00227d8445c41bbe7f5f58e0a226))
* Add shareable URLs for leaderboard tabs ([#194](https://github.com/letta-ai/letta-evals/issues/194)) ([cc61f51](https://github.com/letta-ai/letta-evals/commit/cc61f5143be4071b783a8aab805ba26122274bbb))
* Add token-level data (IDs + logprobs) to TargetResult for training ([#185](https://github.com/letta-ai/letta-evals/issues/185)) ([70950d7](https://github.com/letta-ai/letta-evals/commit/70950d707650942aa840bc0e57a38e90574d85fa))
* Filesystem leaderboard refresh with code agent results ([#188](https://github.com/letta-ai/letta-evals/issues/188)) ([be3a886](https://github.com/letta-ai/letta-evals/commit/be3a8866c34492965c34d45f02abbb0e964a655c))
* Replace agent_loading with agent_created callback for earlier agent_id surfacing ([#190](https://github.com/letta-ai/letta-evals/issues/190)) ([288c544](https://github.com/letta-ai/letta-evals/commit/288c544f94ff8e46d82f3a6d7b3e4325d1415ec6))
* Replace Anthropic prefill trick with structured outputs for rubric grading ([#192](https://github.com/letta-ai/letta-evals/issues/192)) ([1679fee](https://github.com/letta-ai/letta-evals/commit/1679feee3c3f5567f3ac2517a655a230cbc1c47b))
* Structured error reporting in eval results ([#180](https://github.com/letta-ai/letta-evals/issues/180)) ([56bdd6b](https://github.com/letta-ai/letta-evals/commit/56bdd6bf9823fe2ad2284343d3ea2e420a7f961a))


### Bug Fixes

* Capture agent_id on timeout for letta code ([#182](https://github.com/letta-ai/letta-evals/issues/182)) ([a4551c8](https://github.com/letta-ai/letta-evals/commit/a4551c821ff21044c3265b94353e38e2a91de4fb))
* Include .af agent files in pip package ([#193](https://github.com/letta-ai/letta-evals/issues/193)) ([d12cb72](https://github.com/letta-ai/letta-evals/commit/d12cb72bf3a699eaa02c80008ba5ea63166667a5))
* Move metric aggregate update before render in rich progress ([#191](https://github.com/letta-ai/letta-evals/issues/191)) ([d40cb0a](https://github.com/letta-ai/letta-evals/commit/d40cb0a880fcaeb8dd2d6a5948c07f0fa8e06341))
* Report grading errors to progress callback ([#187](https://github.com/letta-ai/letta-evals/issues/187)) ([9584eb7](https://github.com/letta-ai/letta-evals/commit/9584eb7ed35189182cde8197afe153ecbe5ce86d))
* Sonnet 4.6 configs and leaderboard type ([#189](https://github.com/letta-ai/letta-evals/issues/189)) ([c820105](https://github.com/letta-ai/letta-evals/commit/c8201050732a0b9857d7a7e351cb6aaaff2f21e3))
* Support `new_string` in custom fruit grader example ([#198](https://github.com/letta-ai/letta-evals/issues/198)) ([bc9266b](https://github.com/letta-ai/letta-evals/commit/bc9266b653cb0f441717f143f4925d50357121c4))
* Support max_concurrent and output in suite YAML ([#197](https://github.com/letta-ai/letta-evals/issues/197)) ([c838fc0](https://github.com/letta-ai/letta-evals/commit/c838fc0b29af7f84b06dc3252cfd9225c6369868))


### Chores

* Track cached tokens with letta code ([#184](https://github.com/letta-ai/letta-evals/issues/184)) ([360d777](https://github.com/letta-ai/letta-evals/commit/360d777a4e0eb499b5a03012544ecbc8a750bbc3))

## [0.10.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.9.0...letta-evals-v0.10.0) (2026-02-11)


### Features

* Add `agent_script` support to LettaCodeTarget ([#171](https://github.com/letta-ai/letta-evals/issues/171)) ([39cc8bc](https://github.com/letta-ai/letta-evals/commit/39cc8bc5f4339d7878d7a0843ce16be1d0024a7c))
* Add Google Gemini as rubric grading provider ([#179](https://github.com/letta-ai/letta-evals/issues/179)) ([be7ed5f](https://github.com/letta-ai/letta-evals/commit/be7ed5f266cc61d56353ea4cc821951a576ecd84))
* Evaluate letta code agents on filesystem suite ([#167](https://github.com/letta-ai/letta-evals/issues/167)) ([5f18158](https://github.com/letta-ai/letta-evals/commit/5f1815829b8de69d1d48645e25d82018047ad0be))
* Filesystem v2 ([#170](https://github.com/letta-ai/letta-evals/issues/170)) ([04e4260](https://github.com/letta-ai/letta-evals/commit/04e426037061b53fd69c3848f80a12b02226ef50))


### Bug Fixes

* Agent script for letta code ([#176](https://github.com/letta-ai/letta-evals/issues/176)) ([2d80c9b](https://github.com/letta-ai/letta-evals/commit/2d80c9bce573d727eb5edef5c0d84e6f82b1e60f))
* Filesystem cloud reliability with background streaming, retries, timeouts ([#177](https://github.com/letta-ai/letta-evals/issues/177)) ([3f1e0ad](https://github.com/letta-ai/letta-evals/commit/3f1e0adf4669cef3013880dbb5d3610e9f4d4829))
* Improve leaderboard mobile responsiveness and update nav link ([682645b](https://github.com/letta-ai/letta-evals/commit/682645bd429ec300a53cec36f668edf391a685c6))
* Nav links use inherit instead of initial for dark mode ([e0ec1eb](https://github.com/letta-ai/letta-evals/commit/e0ec1ebc6453bd33480fa2c7c187c0a9ea32c095))
* Pass base_url to letta CLI for LettaCodeTarget ([#157](https://github.com/letta-ai/letta-evals/issues/157)) ([56f01c1](https://github.com/letta-ai/letta-evals/commit/56f01c1f48c5cf0fac6b245da3f5305dadc8e3e4))
* Resolve model handles with reasoning / effort level ([#169](https://github.com/letta-ai/letta-evals/issues/169)) ([b1cacd2](https://github.com/letta-ai/letta-evals/commit/b1cacd2df0ba57df6983feeaffc633acd1dc2bf2))
* Soften dark mode palette to match letta.com ([ab76a96](https://github.com/letta-ai/letta-evals/commit/ab76a96f1d430cc21c4466d75338a60e9dc6f6e1))
* Support agent_id for letta_judge grader (Issue [#156](https://github.com/letta-ai/letta-evals/issues/156)) ([#159](https://github.com/letta-ai/letta-evals/issues/159)) ([e3e06ce](https://github.com/letta-ai/letta-evals/commit/e3e06ce0899e7381aafa7c8ae4bbf0d80bbbc007))
* Track tokens even when no costs ([#164](https://github.com/letta-ai/letta-evals/issues/164)) ([405022b](https://github.com/letta-ai/letta-evals/commit/405022b509c0a1a769cdae22290344dc682090d8))
* Update agent judge examples to use gpt-4o-mini ([#172](https://github.com/letta-ai/letta-evals/issues/172)) ([9b8e4ac](https://github.com/letta-ai/letta-evals/commit/9b8e4acaa293261b40cd1bd2cd150956f9af3fcb))
* Update examples to use cloud by default ([#152](https://github.com/letta-ai/letta-evals/issues/152)) ([b343ceb](https://github.com/letta-ai/letta-evals/commit/b343ceb0701286b54500a5c3673870310b540d13))
* Use --new-agent for letta code ([#166](https://github.com/letta-ai/letta-evals/issues/166)) ([2516391](https://github.com/letta-ai/letta-evals/commit/251639172fdf42323d45a8ba4234bd4c4069f33e))


### Chores

* Add Opus 4.6 to leaderboard ([#175](https://github.com/letta-ai/letta-evals/issues/175)) ([c3437b7](https://github.com/letta-ai/letta-evals/commit/c3437b7ef0b5a7d19c229ecceb341ec774cfab07))
* Add rank column, row styling, and provider filter to leaderboard ([#168](https://github.com/letta-ai/letta-evals/issues/168)) ([a9eb17f](https://github.com/letta-ai/letta-evals/commit/a9eb17f53e28b82d0de9b29c30069051d7ae835e))
* Add sandbox attribute for letta code ([#163](https://github.com/letta-ai/letta-evals/issues/163)) ([84c4f5a](https://github.com/letta-ai/letta-evals/commit/84c4f5a5b7719b3cdbe7195d0b1ea1b61d6595dc))
* Retrieve agent state for letta code ([#173](https://github.com/letta-ai/letta-evals/issues/173)) ([210dfa8](https://github.com/letta-ai/letta-evals/commit/210dfa80389b50775bb0aafaf8a2128fa36baffc))
* Retrieve memory blocks with agent state ([#174](https://github.com/letta-ai/letta-evals/issues/174)) ([784dff2](https://github.com/letta-ai/letta-evals/commit/784dff2f3942caab3225d094b24f62453d1d64cd))
* Track usage metrics for letta code ([#165](https://github.com/letta-ai/letta-evals/issues/165)) ([b1bab86](https://github.com/letta-ai/letta-evals/commit/b1bab86d7d2a8d88e980aeedb8a6f1119ea7cd5a))

## [0.9.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.8.1...letta-evals-v0.9.0) (2025-12-23)


### Features

* Add prompt caching ([#135](https://github.com/letta-ai/letta-evals/issues/135)) ([a1b1988](https://github.com/letta-ai/letta-evals/commit/a1b1988948c28007fc0233950ec72007e5ab8188))
* Per-turn evaluations for multi-turn conversations ([#149](https://github.com/letta-ai/letta-evals/issues/149)) ([23fad93](https://github.com/letta-ai/letta-evals/commit/23fad936f67fc8e2e07f6dfa2f711f320998add6))


### Bug Fixes

* Per-turn evaluation render ([#150](https://github.com/letta-ai/letta-evals/issues/150)) ([a40a253](https://github.com/letta-ai/letta-evals/commit/a40a253e7b85eb84b0a8c0d03142bc2a6b8f328b))


### Chores

* Sort eval results by model name and sample ID for consistent ordering ([#148](https://github.com/letta-ai/letta-evals/issues/148)) ([eae891a](https://github.com/letta-ai/letta-evals/commit/eae891a37387480ea78c7e625e865c4b774eab77))

## [0.8.1](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.8.0...letta-evals-v0.8.1) (2025-12-18)


### Chores

* Add Gemini 3 Flash to leaderboard ([#146](https://github.com/letta-ai/letta-evals/issues/146)) ([bd05b8e](https://github.com/letta-ai/letta-evals/commit/bd05b8ebce574f8cfc0381514e8b66a632ae5652))

## [0.8.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.7.0...letta-evals-v0.8.0) (2025-12-16)


### Features

* Add kimi-k2-thinking ([#125](https://github.com/letta-ai/letta-evals/issues/125)) ([a5ca7e7](https://github.com/letta-ai/letta-evals/commit/a5ca7e7257ea9dc2ca4d6f3ba7e9bc34376232ff))
* Add skills suite for leaderboard ([#128](https://github.com/letta-ai/letta-evals/issues/128)) ([d252523](https://github.com/letta-ai/letta-evals/commit/d252523263f01e6a01e63f7e631aaad4511b100f))
* Remove instance level pass/fail metrics ([#127](https://github.com/letta-ai/letta-evals/issues/127)) ([b7e91b6](https://github.com/letta-ai/letta-evals/commit/b7e91b6d529f0a31cf92699d9ee1001618a16d47))


### Bug Fixes

* Fix visualization status bar + mark empty submissions as errors ([#126](https://github.com/letta-ai/letta-evals/issues/126)) ([ffe75d2](https://github.com/letta-ai/letta-evals/commit/ffe75d2fd265a67e5346e5fa21dbc28494426721))
* Google loads `google.svg` ([#134](https://github.com/letta-ai/letta-evals/issues/134)) ([877538c](https://github.com/letta-ai/letta-evals/commit/877538cd7687e38124734d83d2f91819ad0834b9))
* Leaderboard tooltip, spacing ([#129](https://github.com/letta-ai/letta-evals/issues/129)) ([993683f](https://github.com/letta-ai/letta-evals/commit/993683f020b247a0c6b684fba4138140908e2ca4))
* Migrate to 1.0 SDK ([#143](https://github.com/letta-ai/letta-evals/issues/143)) ([08a2854](https://github.com/letta-ai/letta-evals/commit/08a2854502c93178a8bcfa13b16db85e24b1fd29))
* update index.html ([2dda09f](https://github.com/letta-ai/letta-evals/commit/2dda09f6534b82aae9674243810b83433fbe0ba7))


### Chores

* Add leaderboard favicon ([#138](https://github.com/letta-ai/letta-evals/issues/138)) ([313f380](https://github.com/letta-ai/letta-evals/commit/313f3808315384abae61a396cef1b41f412ab75d))
* Add updates tab to leaderboard ([#137](https://github.com/letta-ai/letta-evals/issues/137)) ([44b0255](https://github.com/letta-ai/letta-evals/commit/44b025518936107bf3aeceef4f043226fc021614))
* GPT 5.2 leaderboard evals ([#139](https://github.com/letta-ai/letta-evals/issues/139)) ([00894cb](https://github.com/letta-ai/letta-evals/commit/00894cb926a45f5085f6f75d708f421d5e3583af))
* GPT 5.2 xhigh leaderboard evals ([#141](https://github.com/letta-ai/letta-evals/issues/141)) ([7f7c4cb](https://github.com/letta-ai/letta-evals/commit/7f7c4cbc8a9efbeb6e7282423e45b7717e12cff5))
* GPT5.1, Gemini 3, Opus 4.5 updates ([#133](https://github.com/letta-ai/letta-evals/issues/133)) ([de1f4ca](https://github.com/letta-ai/letta-evals/commit/de1f4ca6c01065c3020f64430d4d2125cecc386e))
* Improve empty submission check ([#132](https://github.com/letta-ai/letta-evals/issues/132)) ([2463a5b](https://github.com/letta-ai/letta-evals/commit/2463a5bf7616b4ca735bfa73f31b271b303ed5f2))
* Multiple benchmarks on leaderboard ([#124](https://github.com/letta-ai/letta-evals/issues/124)) ([39cb733](https://github.com/letta-ai/letta-evals/commit/39cb733e08f8896c634e56025ac2aa81b6cd7d70))
* Track costs during runs ([#130](https://github.com/letta-ai/letta-evals/issues/130)) ([dc980a1](https://github.com/letta-ai/letta-evals/commit/dc980a1d2744eb66e8c095ddc617562f17c07e2c))
* Update leaderboard logs ([#142](https://github.com/letta-ai/letta-evals/issues/142)) ([e34e0cd](https://github.com/letta-ai/letta-evals/commit/e34e0cd7d12d0869f118ef4d573c7028370db299))
* Update leaderboard with Deepseek v3.2 and Mistral large 3 ([#136](https://github.com/letta-ai/letta-evals/issues/136)) ([01a17cb](https://github.com/letta-ai/letta-evals/commit/01a17cb0ac1cf0ca7eb5040328675fba44815297))

## [0.7.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.6.1...letta-evals-v0.7.0) (2025-11-04)


### Features

* Evaluate multiple models with letta-code ([#113](https://github.com/letta-ai/letta-evals/issues/113)) ([f98933b](https://github.com/letta-ai/letta-evals/commit/f98933b0c0ac2a6a20db36ecc4c12941ba71f0cd))
* Support multiple graders in gates with weighted average and logical combinations ([#117](https://github.com/letta-ai/letta-evals/issues/117)) ([d0d0add](https://github.com/letta-ai/letta-evals/commit/d0d0add8199220746cf9f678f0f3357c5ae4a90e))


### Bug Fixes

* Fix per sample pass is None ([#120](https://github.com/letta-ai/letta-evals/issues/120)) ([f416487](https://github.com/letta-ai/letta-evals/commit/f416487407abfa6a38b62b98a4e6a957654eb018))


### Documentation

* Add examples for multiple metric gates ([#118](https://github.com/letta-ai/letta-evals/issues/118)) ([739e4de](https://github.com/letta-ai/letta-evals/commit/739e4de994ebea5551b0dcf6f83da57a151f51ab))
* Add reference to multi grader gate example in top level README ([#119](https://github.com/letta-ai/letta-evals/issues/119)) ([7ad732d](https://github.com/letta-ai/letta-evals/commit/7ad732dd2db20842fde523ad5fae4b2bb09be8db))
* Clean up README of Claude references ([#114](https://github.com/letta-ai/letta-evals/issues/114)) ([a41f9d0](https://github.com/letta-ai/letta-evals/commit/a41f9d039b230dd78337ef043650471455ddfbbb))

## [0.6.1](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.6.0...letta-evals-v0.6.1) (2025-10-30)


### Bug Fixes

* Letta code session_id to agent_id ([#112](https://github.com/letta-ai/letta-evals/issues/112)) ([3ad7c0c](https://github.com/letta-ai/letta-evals/commit/3ad7c0c127c34b8e80fa34ed0df686d1687b94f6))
* patch site ([24cfa0b](https://github.com/letta-ai/letta-evals/commit/24cfa0bfc6d231cb4fe58b221630782a6b2c00c0))


### Chores

* fix lints ([#111](https://github.com/letta-ai/letta-evals/issues/111)) ([e3431b8](https://github.com/letta-ai/letta-evals/commit/e3431b82455345021c5d2ffe6c7f0a38a67f51a9))
* fix styling ([7dd2818](https://github.com/letta-ai/letta-evals/commit/7dd2818300d20d31b4b1d4ab9810e00b74b20add))

## [0.6.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.5.0...letta-evals-v0.6.0) (2025-10-29)


### Features

* add eval website ([#104](https://github.com/letta-ai/letta-evals/issues/104)) ([2daaf0c](https://github.com/letta-ai/letta-evals/commit/2daaf0c598780b6fa2edc26de52348de127f88a5))
* Fix workflow ([#105](https://github.com/letta-ai/letta-evals/issues/105)) ([62207bd](https://github.com/letta-ai/letta-evals/commit/62207bd502fd6e99008ac755e7956bc3cfa42053))
* Support letta code as builtin target ([#101](https://github.com/letta-ai/letta-evals/issues/101)) ([fe1ae2f](https://github.com/letta-ai/letta-evals/commit/fe1ae2f7a1bc25ead8115459eda148c873c5ef27))


### Bug Fixes

* Fix kwargs in run function ([#97](https://github.com/letta-ai/letta-evals/issues/97)) ([c0f64b0](https://github.com/letta-ai/letta-evals/commit/c0f64b0a317f7c70675e10ee02ed47b19efc09a7))
* Remove duplicate gpt 4.1 results ([#95](https://github.com/letta-ai/letta-evals/issues/95)) ([2b9092c](https://github.com/letta-ai/letta-evals/commit/2b9092cd7b4180f493a5acfd0e0c33662ae574c2))
* Update Sonnet 4.5 cost ([#96](https://github.com/letta-ai/letta-evals/issues/96)) ([3b7a39d](https://github.com/letta-ai/letta-evals/commit/3b7a39dd85f02412b644c0f65162d6f52a772921))


### Refactors

* Add extra vars to Sample ([#100](https://github.com/letta-ai/letta-evals/issues/100)) ([9ce87cd](https://github.com/letta-ai/letta-evals/commit/9ce87cda70e167533b8d27763d53aad844e63313))
* Make target spec a discriminated union ([#103](https://github.com/letta-ai/letta-evals/issues/103)) ([762b520](https://github.com/letta-ai/letta-evals/commit/762b520313e57107a756744f2990b325a4478d14))
* Refactor AgentTarget to LettaAgentTarget ([#98](https://github.com/letta-ai/letta-evals/issues/98)) ([952d6f8](https://github.com/letta-ai/letta-evals/commit/952d6f80d50f6547ab557d8f1d6e445dd8364186))
* Refactor Target to AbstractAgentTarget ([#99](https://github.com/letta-ai/letta-evals/issues/99)) ([bb632e0](https://github.com/letta-ai/letta-evals/commit/bb632e096607ad47f61ef83f9a64a98d966b964e))


### Documentation

* Add letta code example to READMEs ([#102](https://github.com/letta-ai/letta-evals/issues/102)) ([2fe0f68](https://github.com/letta-ai/letta-evals/commit/2fe0f680106a7593543f7c283d8cc83671a0af67))
* adjust ([2b16cd3](https://github.com/letta-ai/letta-evals/commit/2b16cd390378ea473d1f07690d9bd435b603ed04))
* patch svg, update site ([64c3491](https://github.com/letta-ai/letta-evals/commit/64c34913e056a037af770b1d8ecc8c99ae1512ad))


### Chores

* Add leaderboard results yaml and script ([#93](https://github.com/letta-ai/letta-evals/issues/93)) ([e24fb74](https://github.com/letta-ai/letta-evals/commit/e24fb74c373b19cf3fc7543d365f07de6b547ce7))
* fix style ([#108](https://github.com/letta-ai/letta-evals/issues/108)) ([575be53](https://github.com/letta-ai/letta-evals/commit/575be5373d6362583f2c5c8bfeb5753835ff5a14))
* relative routes bruh ([#109](https://github.com/letta-ai/letta-evals/issues/109)) ([d68ee95](https://github.com/letta-ai/letta-evals/commit/d68ee95cdde199381148c8e678eca71a8eb1e334))
* whoops ([#107](https://github.com/letta-ai/letta-evals/issues/107)) ([53fc4eb](https://github.com/letta-ai/letta-evals/commit/53fc4eb4375d91e836b257c92462987b5f0fd07c))

## [0.5.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.4.1...letta-evals-v0.5.0) (2025-10-23)


### Features

* Add agent_id to visualization ([#91](https://github.com/letta-ai/letta-evals/issues/91)) ([2f77348](https://github.com/letta-ai/letta-evals/commit/2f77348458a90141f984b7c15274dc9f6d7ffdaf))
* Add agent-as-judge support for rubric grading ([#77](https://github.com/letta-ai/letta-evals/issues/77)) ([ae4878e](https://github.com/letta-ai/letta-evals/commit/ae4878e067986b989fce6fd77b281b07345e822e))
* Add summary tables on suite finish for all display types (rich, simple) ([#92](https://github.com/letta-ai/letta-evals/issues/92)) ([19e1e1c](https://github.com/letta-ai/letta-evals/commit/19e1e1c8be3c8edbc8b16474353f2be5891f2313))
* Support anthropic models as grader ([#83](https://github.com/letta-ai/letta-evals/issues/83)) ([c38cf1f](https://github.com/letta-ai/letta-evals/commit/c38cf1f4c5ae80fcc4838eb1878d3f735222506d))
* Support default Letta judge agent with new `letta_judge` grader kind ([#86](https://github.com/letta-ai/letta-evals/issues/86)) ([b4bfd6c](https://github.com/letta-ai/letta-evals/commit/b4bfd6cf26514c9a139a88d632bc35b35b9949b9))


### Bug Fixes

* Add defensive check for run_id from streaming chunk ([#75](https://github.com/letta-ai/letta-evals/issues/75)) ([7d34884](https://github.com/letta-ai/letta-evals/commit/7d348843f66f418bafaceb130354af4ec579fff5))
* Add pre-fill trick for Anthropic json output ([#84](https://github.com/letta-ai/letta-evals/issues/84)) ([2a4fd4a](https://github.com/letta-ai/letta-evals/commit/2a4fd4aadda3fe4f839e70d3393d9da439911ee2))
* Fix retry logic for failing agent ([#74](https://github.com/letta-ai/letta-evals/issues/74)) ([ecd5d5a](https://github.com/letta-ai/letta-evals/commit/ecd5d5abdbc6b4edea0651e6e4bc2f7ee07218c0))
* Fix typo in chunks appending ([#82](https://github.com/letta-ai/letta-evals/issues/82)) ([d888474](https://github.com/letta-ai/letta-evals/commit/d8884742477e28bdc4b089295fce22f4f053c19c))
* OpenRouter for Kimi ([#76](https://github.com/letta-ai/letta-evals/issues/76)) ([2cd6192](https://github.com/letta-ai/letta-evals/commit/2cd6192b326e6c09b7600726dad45d4f5a246d42))
* Print out chunks on run_id error ([#81](https://github.com/letta-ai/letta-evals/issues/81)) ([8b71a64](https://github.com/letta-ai/letta-evals/commit/8b71a6435adff3e8505780da332553a0dbbf3a4a))


### Performance Improvements

* Retry on letta server failures ([#72](https://github.com/letta-ai/letta-evals/issues/72)) ([acaf5a5](https://github.com/letta-ai/letta-evals/commit/acaf5a5f152c48d0902d3c0696f248d5686ad0c0))


### Refactors

* Flatten package imports for easier pip usage ([#89](https://github.com/letta-ai/letta-evals/issues/89)) ([24fd61a](https://github.com/letta-ai/letta-evals/commit/24fd61a5c47cb666e886f1e03bf018d072a5f098))
* Rename `rubric` to `model_judge` ([#87](https://github.com/letta-ai/letta-evals/issues/87)) ([0047d3f](https://github.com/letta-ai/letta-evals/commit/0047d3f8fa1b7c556ffefecfd9123736b703d541))
* Use Pydantic discriminated union for GraderSpec types ([#88](https://github.com/letta-ai/letta-evals/issues/88)) ([bb21f1b](https://github.com/letta-ai/letta-evals/commit/bb21f1b9469ada4b87779e40ce8fbbbed663a27d))


### Documentation

* Add perma-link for UI screenshot ([#79](https://github.com/letta-ai/letta-evals/issues/79)) ([00dfb38](https://github.com/letta-ai/letta-evals/commit/00dfb38eba43d90ef4e3f6aae448e5b5c6501417))
* Fix typo in README ([#80](https://github.com/letta-ai/letta-evals/issues/80)) ([a5e0497](https://github.com/letta-ai/letta-evals/commit/a5e049767d22c16a64b62dea6a41577c80c690c0))
* Write multi-turn memory example ([#90](https://github.com/letta-ai/letta-evals/issues/90)) ([a683727](https://github.com/letta-ai/letta-evals/commit/a68372792eef405b1d7fd874c33cb2f3440cc4b0))

## [0.4.1](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.4.0...letta-evals-v0.4.1) (2025-10-21)


### Features

* Add `ruff check --fix .` to README ([#65](https://github.com/letta-ai/letta-evals/issues/65)) ([066408b](https://github.com/letta-ai/letta-evals/commit/066408b99151caa8edf684d24540168067b45e2c))


### Bug Fixes

* Use rubric grader for file system task ([#63](https://github.com/letta-ai/letta-evals/issues/63)) ([03ef537](https://github.com/letta-ai/letta-evals/commit/03ef5370c9223cfa03377c83e13376d00503f8d8))


### Documentation

* Add example for tool output extractor ([#66](https://github.com/letta-ai/letta-evals/issues/66)) ([523d498](https://github.com/letta-ai/letta-evals/commit/523d498ff77c8f4a13239a124d3bf13a3f8b4ebd))
* Add extra instructions for cloud usage ([#71](https://github.com/letta-ai/letta-evals/issues/71)) ([a8d3e5b](https://github.com/letta-ai/letta-evals/commit/a8d3e5bf674e46296c4ac520405d06fd9f978e07))
* Write high level README ([#69](https://github.com/letta-ai/letta-evals/issues/69)) ([68be3da](https://github.com/letta-ai/letta-evals/commit/68be3da0fa3c3cf776c716fe48fc9014d5849de8))


### Chores

* Update preview image ([#70](https://github.com/letta-ai/letta-evals/issues/70)) ([81d5b41](https://github.com/letta-ai/letta-evals/commit/81d5b418c9d37e061b89e1ac20b1fda1be31aa37))

## [0.4.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.3.2...letta-evals-v0.4.0) (2025-10-21)


### Features

* Add multi-run ([#57](https://github.com/letta-ai/letta-evals/issues/57)) ([23ead2d](https://github.com/letta-ai/letta-evals/commit/23ead2d8f1f2a98bbd26aa7ad42c230e1c47d156))
* Add rubric vars ([#58](https://github.com/letta-ai/letta-evals/issues/58)) ([29f775a](https://github.com/letta-ai/letta-evals/commit/29f775a0324329885e73e093b43b6a91cb55795f))
* Support csv loading ([#60](https://github.com/letta-ai/letta-evals/issues/60)) ([28bf491](https://github.com/letta-ai/letta-evals/commit/28bf49181dc3071f4012c1e65e0ef00e9e858669))


### Bug Fixes

* Update filesystem dataset ([#62](https://github.com/letta-ai/letta-evals/issues/62)) ([634bfc6](https://github.com/letta-ai/letta-evals/commit/634bfc6be0076373dc3ebeb4cbe7376b721d8486))


### Chores

* Replace deepseek and GLM models ([#61](https://github.com/letta-ai/letta-evals/issues/61)) ([3f0bf79](https://github.com/letta-ai/letta-evals/commit/3f0bf7914fc0660ef05ac0ab04b894fc8c2fb043))

## [0.3.2](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.3.1...letta-evals-v0.3.2) (2025-10-20)


### Features

* Add visualization library and simple visualization configurations ([#55](https://github.com/letta-ai/letta-evals/issues/55)) ([57d483d](https://github.com/letta-ai/letta-evals/commit/57d483d899de6f82e8557b8cefa439750c656bfb))

## [0.3.1](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.3.0...letta-evals-v0.3.1) (2025-10-20)


### Bug Fixes

* Package python files in build ([#53](https://github.com/letta-ai/letta-evals/issues/53)) ([d5e2682](https://github.com/letta-ai/letta-evals/commit/d5e26824e49a8783f4357117a1d646244fcdd911))

## [0.3.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.2.0...letta-evals-v0.3.0) (2025-10-20)


### Features

* Add filesystem benchmark generator ([#48](https://github.com/letta-ai/letta-evals/issues/48)) ([c2be72a](https://github.com/letta-ai/letta-evals/commit/c2be72ae2704c9e42eeeeaf5e319287a79d405ce))
* Add max samples to display ([#47](https://github.com/letta-ai/letta-evals/issues/47)) ([c821171](https://github.com/letta-ai/letta-evals/commit/c821171b539233119c28a095e951a82990442b4c))
* Add memory block built-in extractor  ([#51](https://github.com/letta-ai/letta-evals/issues/51)) ([d171a10](https://github.com/letta-ai/letta-evals/commit/d171a10b662461ac8c07dd39f01d2ae40b7ffda4))
* Remove hardcoding metric to accuracy ([#45](https://github.com/letta-ai/letta-evals/issues/45)) ([62f41c4](https://github.com/letta-ai/letta-evals/commit/62f41c4847fa8c3ee4a5294776e81cd13748923c))
* Support passing in handles instead of just model configs ([#41](https://github.com/letta-ai/letta-evals/issues/41)) ([927e70a](https://github.com/letta-ai/letta-evals/commit/927e70ae3a163094cf73c2aee48ec2e691bdd3a6))
* Use `gpt-5-mini` as rubric grader model ([#46](https://github.com/letta-ai/letta-evals/issues/46)) ([4a4a0ff](https://github.com/letta-ai/letta-evals/commit/4a4a0ff8b4cd4e689775fe8fd59f2d459d976723))


### Bug Fixes

* Cannot access local variable 'stream' error  ([#33](https://github.com/letta-ai/letta-evals/issues/33)) ([a989a16](https://github.com/letta-ai/letta-evals/commit/a989a16e09675a850474b7e9c6e3f042e5822835))
* Expunge send_message and disable tool rules ([7775596](https://github.com/letta-ai/letta-evals/commit/77755964f3c01578bfe52dfe5600d30534ef86d8))
* Fix streaming bug returns partial results ([8d3e3d8](https://github.com/letta-ai/letta-evals/commit/8d3e3d8abec16aebb9867605b778a4ffa9ce3145))
* Model, status and metric columns after evaluation completes ([#34](https://github.com/letta-ai/letta-evals/issues/34)) ([b47c508](https://github.com/letta-ai/letta-evals/commit/b47c508ead27a98aceb349470e644f80387defeb))
* Update leaderboard task suites ([#35](https://github.com/letta-ai/letta-evals/issues/35)) ([436ce6f](https://github.com/letta-ai/letta-evals/commit/436ce6fd8628de37da1815f4a63928f23d2037a2))


### Refactors

* Support passing in token, base_url, and project_id programmatically ([#36](https://github.com/letta-ai/letta-evals/issues/36)) ([1e3780a](https://github.com/letta-ai/letta-evals/commit/1e3780af91b3f02dd5b2930d1b2c7480375eccaa))


### Documentation

* Add README for memory block extraction ([#52](https://github.com/letta-ai/letta-evals/issues/52)) ([49ccf25](https://github.com/letta-ai/letta-evals/commit/49ccf25c006eace82872e1e5c0e16739bd676be2))


### Chores

* Configurable retries and timeout ([9ee97b6](https://github.com/letta-ai/letta-evals/commit/9ee97b6c5e7399b37321829685a37730670bed5f))
* Report average metrics across attempted and total samples ([#50](https://github.com/letta-ai/letta-evals/issues/50)) ([f2b4f7a](https://github.com/letta-ai/letta-evals/commit/f2b4f7ad1f3bbe8fd266d3b08e9501783f80bb75))
* Separate files for headers and summary ([#49](https://github.com/letta-ai/letta-evals/issues/49)) ([9f3dbcc](https://github.com/letta-ai/letta-evals/commit/9f3dbccc2516a6730ef64a511f7ac54656c46e60))
* Update examples to use `letta_v1_agent` ([#31](https://github.com/letta-ai/letta-evals/issues/31)) ([62a6ab6](https://github.com/letta-ai/letta-evals/commit/62a6ab626816d28220c474c00c634b1cfc9e66dc))
* Update model configs ([91afcb8](https://github.com/letta-ai/letta-evals/commit/91afcb81e48537ad07bbb4dff11af2648ccae6e2))

## [0.2.0](https://github.com/letta-ai/letta-evals/compare/letta-evals-v0.1.0...letta-evals-v0.2.0) (2025-10-15)


### Features

* Add builtin tool output/arguments extractors ([5133966](https://github.com/letta-ai/letta-evals/commit/51339668f797232e9e5119750e1b243feeb37912))
* add core memory update benchmark ([#21](https://github.com/letta-ai/letta-evals/issues/21)) ([e0261de](https://github.com/letta-ai/letta-evals/commit/e0261dea683c0f52abc10ea4aeb767ee71258e9c))
* add filesystem eval ([#24](https://github.com/letta-ai/letta-evals/issues/24)) ([55b902d](https://github.com/letta-ai/letta-evals/commit/55b902d474dee79d1b5203ae077c83ad6fe925b6))
* add letta leaderboard ([#8](https://github.com/letta-ai/letta-evals/issues/8)) ([ae68e22](https://github.com/letta-ai/letta-evals/commit/ae68e2267c6a89ccb298c5f190960b93b357ec01))
* Add model configs and multi-model runners ([cf6a707](https://github.com/letta-ai/letta-evals/commit/cf6a707c25e9bbbf1c6c9e827ec61a15e62df84a))
* Add programmatic agent creation ([fd820a2](https://github.com/letta-ai/letta-evals/commit/fd820a23d6ce0383ae128ec0fa6f24241fd47099))
* Add support for re-grading cached evaluation trajectories ([#19](https://github.com/letta-ai/letta-evals/issues/19)) ([8feab64](https://github.com/letta-ai/letta-evals/commit/8feab6442dbd331c8a9abd9158f471e576302ac0))
* Clean up results.json schema ([#18](https://github.com/letta-ai/letta-evals/issues/18)) ([ec72e2b](https://github.com/letta-ai/letta-evals/commit/ec72e2b9c077b0684b318a064d8b4d8463c28074))
* Flatten directories further ([759638c](https://github.com/letta-ai/letta-evals/commit/759638c5297047641c6bb9af8dc4a8f655d7add6))
* Implement decorator based custom functions ([e53ee19](https://github.com/letta-ai/letta-evals/commit/e53ee199a2a44b9d859c9f6a3c768beb49ce7058))
* Refactor to use TaskGroups ([9bc810b](https://github.com/letta-ai/letta-evals/commit/9bc810b66dba61e2ac4794addd2cbd3647a256e7))
* Support custom extractors/Python tool evaluators ([fc1b2f7](https://github.com/letta-ai/letta-evals/commit/fc1b2f7b375ed0e7d3666a2fe0746ed92b9f27a7))
* support multiple metrics ([#27](https://github.com/letta-ai/letta-evals/issues/27)) ([b0fa023](https://github.com/letta-ai/letta-evals/commit/b0fa02357733451d0c216a1cb36882556f84d9e2))
* Support relative paths for custom graders ([6f3cda3](https://github.com/letta-ai/letta-evals/commit/6f3cda39eff447477b81020b73985ae6cf25af5b))
* Support streaming for stability ([ef18ef6](https://github.com/letta-ai/letta-evals/commit/ef18ef63c3986cecd48aa15bc261c04ddca94af0))
* update together configs ([#20](https://github.com/letta-ai/letta-evals/issues/20)) ([79d7890](https://github.com/letta-ai/letta-evals/commit/79d7890a758b7b97162ed91aca3f752d2280df28))


### Chores

* Prepare repo for Pypi publishing ([#28](https://github.com/letta-ai/letta-evals/issues/28)) ([6ad92b2](https://github.com/letta-ai/letta-evals/commit/6ad92b2477a549065a0885f9985bab8202e16906))
* Rename ideal to ground truth ([9367c05](https://github.com/letta-ai/letta-evals/commit/9367c05599b98da8279474a0007bd664a18730fd))
