import argparse
import sys

from cli.run import run_application
from cli.benchmark import run_benchmark
from cli.diagnostics import run_diagnostics
from cli.codesign import run_codesign

def main():
    """
    Main entry point for the cim-sim CLI interface.
    Parses arguments and dispatches to the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(
        description="Organic CIM Simulation & Neuromorphic Computing Platform CLI Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run a specific CIM bionic application")
    run_parser.add_argument(
        "app", 
        help="Name of the neuromorphic application to run.\n"
             "Options: ecg_cardio, fatigue_eeg, bearing_fault, chaotic_lorenz, "
             "digit_rec, speech_emotion, embodied_ai, edge_llm, physical_attention, "
             "fingerprint_rec, cifar10_vision, optoelectronic_vision, neuromorphic_stdp, "
             "neuromorphic_pid, tactile_eskin, neuromorphic_grasp, seizure_detection, "
             "biohybrid_spiking, face_rec, dvs_gesture, ppg_heartrate, neuromorphic_rl, text_sentiment, "
             "olfactory_enose, eeg_motor_imagery, neuromorphic_kws"
    )
    
    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a bionic device across multiple applications")
    bench_parser.add_argument("--device", required=True, help="Name or filepath of custom device profile JSON")
    bench_parser.add_argument("--apps", help="Comma-separated application names to test (e.g. fingerprint_rec,ecg_cardio)")
    bench_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per benchmark item (default: 3)")

    # Subcommand: codesign
    codesign_parser = subparsers.add_parser("codesign", help="Run co-design compilation and self-healing verification on a device profile")
    codesign_parser.add_argument("--device", required=True, help="Name or filepath of custom device profile JSON")

    # Subcommand: diagnose
    diagnose_parser = subparsers.add_parser("diagnose", help="Generate physical diagnostic curves and datasheet report for a device profile")
    diagnose_parser.add_argument("--device", required=True, help="Name or filepath of custom device profile JSON")

    # Subcommand: publish
    publish_parser = subparsers.add_parser("publish", help="Run the top-journal comparative benchmark and publish reports")

    # Subcommand: prepare-data
    prepare_parser = subparsers.add_parser("prepare-data", help="Download standard datasets and generate lightweight mock data for custom sensors")

    args, extra_args = parser.parse_known_args()
    
    # Fallback to standard app run if subcommands are not specified
    if not args.command:
        # If user did e.g. "python main.py digit_rec", map to subcommand run
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            args.command = "run"
            args.app = sys.argv[1]
            extra_args = sys.argv[2:]
        else:
            parser.print_help()
            print("\n💡 Example usage:")
            print("  python main.py run digit_rec")
            print("  python main.py benchmark --device FingerMemristor --epochs 3")
            print("  python main.py codesign --device FingerMemristor")
            print("  python main.py diagnose --device FingerMemristor")
            print("  python main.py publish")
            return
            
    if args.command == "run":
        run_application(args.app, extra_args)
    elif args.command == "benchmark":
        apps_list = []
        if args.apps:
            apps_list = [a.strip() for a in args.apps.split(",")]
        run_benchmark(args.device, apps_list, args.epochs)
    elif args.command == "codesign":
        run_codesign(args.device)
    elif args.command == "diagnose":
        run_diagnostics(args.device)
    elif args.command == "publish":
        from scripts.run_top_journal_benchmark import main as run_publish
        run_publish()
    elif args.command == "prepare-data":
        from scripts.download_datasets import download_all_datasets
        download_all_datasets()

if __name__ == "__main__":
    main()
