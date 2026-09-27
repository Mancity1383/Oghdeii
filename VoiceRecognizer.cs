using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Speech.Recognition;

namespace LaptopTap
{
    class Program
    {
        // Set while Main is deliberately tearing the engine down, so the
        // RecognizeCompleted handler only treats *unexpected* completions
        // (mic unplugged, device locked, engine wedged) as fatal.
        static volatile bool shuttingDown = false;

        static RecognizerInfo ChooseEnglishRecognizer()
        {
            List<RecognizerInfo> installed = SpeechRecognitionEngine.InstalledRecognizers().ToList();
            RecognizerInfo exact = installed.FirstOrDefault(r =>
                r.Culture != null && r.Culture.Name.Equals("en-US", StringComparison.OrdinalIgnoreCase));
            if (exact != null) return exact;
            return installed.FirstOrDefault(r =>
                r.Culture != null && r.Culture.TwoLetterISOLanguageName.Equals("en", StringComparison.OrdinalIgnoreCase));
        }

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("STATUS:STARTING");
            Console.Out.Flush();

            // args[0] remains the legacy confidence setting for compatibility;
            // Python applies the minimum-confidence gate before contacting V1M.
            bool requireWakeWord = args.Length < 2 ||
                !args[1].Equals("direct", StringComparison.OrdinalIgnoreCase);

            SpeechRecognitionEngine engine = null;
            try
            {
                RecognizerInfo recognizer = ChooseEnglishRecognizer();
                if (recognizer == null)
                    throw new InvalidOperationException(
                        "No English Windows speech recognizer is installed. Install an English speech language in Windows Settings."
                    );

                Console.WriteLine("STATUS:RECOGNIZER:" + recognizer.Culture.Name + ":" + recognizer.Name);
                Console.Out.Flush();

                try
                {
                    engine = new SpeechRecognitionEngine(recognizer.Id);
                }
                catch (ArgumentException)
                {
                    Console.WriteLine("STATUS:RECOGNIZER_ID_FALLBACK:" + recognizer.Culture.Name);
                    Console.Out.Flush();
                    try
                    {
                        engine = new SpeechRecognitionEngine(recognizer.Culture);
                    }
                    catch (ArgumentException)
                    {
                        engine = new SpeechRecognitionEngine();
                    }
                }
                engine.SetInputToDefaultAudioDevice();
                engine.MaxAlternates = 5;

                // Constrained recognition restores the strong accuracy of the
                // original offline mode: Windows chooses among supported
                // phrases instead of unconstrained dictation. The selected
                // phrase and its N-best alternatives still go to V1M, which
                // alone decides validity and the action to run.
                string[] directPhrases = new string[] {
                    "copy", "copy that", "copy this",
                    "paste", "paste that", "paste this",
                    "screenshot", "take screenshot", "take a screenshot",
                    "lock", "lock pc", "lock screen", "lock computer",
                    "undo", "undo that", "redo", "select all",
                    "desktop", "show desktop", "show the desktop",
                    "calculator", "open calculator",
                    "notepad", "open notepad",
                    "browser", "open browser",
                    "terminal", "open terminal",
                    "close window", "close app",
                    "play", "play music", "pause", "pause music",
                    "next", "next track", "previous", "previous track",
                    "mute", "mute sound",
                    "volume up", "turn volume up",
                    "volume down", "turn volume down"
                };
                string[] commandPhrases;
                if (requireWakeWord)
                {
                    string[] wakePrefixes = new string[] { "laptop ", "lap top ", "hey laptop " };
                    commandPhrases = wakePrefixes
                        .SelectMany(prefix => directPhrases.Select(phrase => prefix + phrase))
                        .ToArray();
                }
                else
                {
                    commandPhrases = directPhrases;
                }
                Choices choices = new Choices(commandPhrases);
                GrammarBuilder commandBuilder = new GrammarBuilder();
                commandBuilder.Culture = recognizer.Culture;
                commandBuilder.Append(choices);
                Grammar commandGrammar = new Grammar(commandBuilder)
                {
                    Name = "LaptopTapCommands",
                    Priority = 1,
                    Weight = 1.0f
                };
                engine.LoadGrammar(commandGrammar);

                // End the utterance after a brief pause, while allowing a
                // natural gap between the wake phrase and the command.
                engine.InitialSilenceTimeout = TimeSpan.FromMilliseconds(0);
                engine.BabbleTimeout = TimeSpan.FromMilliseconds(0);
                engine.EndSilenceTimeout = TimeSpan.FromMilliseconds(400);
                engine.EndSilenceTimeoutAmbiguous = TimeSpan.FromMilliseconds(550);

                engine.SpeechDetected += (s, e) =>
                {
                    Console.WriteLine("SPEECH_ACTIVE");
                    Console.Out.Flush();
                };

                string lastTranscript = null;
                DateTime lastAcceptedAt = DateTime.MinValue;
                object debounceLock = new object();

                engine.SpeechRecognized += (s, e) =>
                {
                    if (e.Result == null) return;
                    float conf = e.Result.Confidence;
                    string text = (e.Result.Text ?? "").Trim()
                        .Replace('\r', ' ').Replace('\n', ' ');
                    if (String.IsNullOrWhiteSpace(text)) return;
                    string transcriptKey = text.ToLowerInvariant();

                    bool duplicate = false;
                    lock (debounceLock)
                    {
                        DateTime now = DateTime.UtcNow;
                        if (transcriptKey == lastTranscript && (now - lastAcceptedAt).TotalMilliseconds < 450)
                            duplicate = true;
                        else
                        {
                            lastTranscript = transcriptKey;
                            lastAcceptedAt = now;
                        }
                    }
                    if (!duplicate)
                    {
                        List<string> candidates = new List<string>();
                        HashSet<string> seenCandidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                        Action<string, float> addCandidate = (candidateText, candidateConfidence) =>
                        {
                            candidateText = (candidateText ?? "").Trim()
                                .Replace('\r', ' ').Replace('\n', ' ');
                            if (String.IsNullOrWhiteSpace(candidateText) ||
                                !seenCandidates.Add(candidateText)) return;
                            candidates.Add(
                                candidateConfidence.ToString("F3", CultureInfo.InvariantCulture) +
                                "\u001e" + candidateText
                            );
                        };
                        addCandidate(text, conf);
                        foreach (RecognizedPhrase alternate in e.Result.Alternates.Take(4))
                            addCandidate(alternate.Text, alternate.Confidence);

                        string candidatePayload = String.Join("\u001f", candidates.ToArray());
                        string encodedCandidates = Convert.ToBase64String(
                            Encoding.UTF8.GetBytes(candidatePayload)
                        );
                        Console.WriteLine(
                            "VOICE_TEXT2:" +
                            conf.ToString("F3", CultureInfo.InvariantCulture) + ":" +
                            encodedCandidates
                        );
                        Console.Out.Flush();
                    }
                };

                engine.SpeechRecognitionRejected += (s, e) =>
                {
                    if (e.Result == null) return;
                    float conf = e.Result.Confidence;
                    string text = (e.Result.Text ?? "").ToLowerInvariant().Trim();
                    Console.WriteLine(
                        "DEBUG_REJECTED_TEXT:" +
                        conf.ToString("F3", CultureInfo.InvariantCulture) + ":" + text
                    );
                    Console.Out.Flush();
                };

                engine.RecognizeCompleted += (s, e) =>
                {
                    if (shuttingDown) return;
                    string detail = e.Error != null
                        ? e.Error.Message
                        : "recognition pipeline ended unexpectedly";
                    Console.WriteLine("ERROR:Recognition completed with error: " + detail);
                    Console.Out.Flush();
                    // In Multiple mode completion means the engine can no
                    // longer hear us (device removed, session locked...).
                    // Exit non-zero so the Python watchdog respawns us and
                    // re-attaches to the default microphone instead of
                    // leaving a live-but-deaf process behind.
                    Environment.Exit(1);
                };

                Console.WriteLine("STATUS:MODE:CONSTRAINED_" + (requireWakeWord ? "WAKE" : "DIRECT"));
                Console.WriteLine("STATUS:READY");
                Console.Out.Flush();
                engine.RecognizeAsync(RecognizeMode.Multiple);

                string line;
                while ((line = Console.ReadLine()) != null)
                {
                    string command = line.Trim().ToLowerInvariant();
                    if (command == "quit" || command == "exit") break;
                    // Liveness probe used by the Python monitor thread: a
                    // helper that stops answering is treated as hung.
                    if (command == "ping")
                    {
                        Console.WriteLine("PONG");
                        Console.Out.Flush();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("ERROR:" + ex.GetType().Name + ": " + ex.Message);
                Console.Out.Flush();
                Environment.ExitCode = 1;
            }
            finally
            {
                shuttingDown = true;
                if (engine != null)
                {
                    try { engine.RecognizeAsyncCancel(); } catch { }
                    try { engine.Dispose(); } catch { }
                }
                Console.WriteLine("STATUS:STOPPED");
                Console.Out.Flush();
            }
        }
    }
}
