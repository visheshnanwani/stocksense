import unittest
from threading import Event
from training_jobs import TrainingJobs


class TrainingJobTests(unittest.TestCase):
    def test_reconnecting_observer_reuses_running_and_completed_job(self):
        jobs = TrainingJobs()
        started, finish = Event(), Event()
        calls = []
        def train(report):
            calls.append(1)
            report('Working')
            started.set()
            finish.wait(5)
            return 'model'
        try:
            first = jobs.get_or_start('IDEA.NS', train)
            self.assertTrue(started.wait(2))
            second = jobs.get_or_start('IDEA.NS', train)
            self.assertIs(first, second)
            self.assertEqual(second.stage, 'Working')
            finish.set()
            self.assertEqual(second.future.result(2), 'model')
            self.assertIs(jobs.get_or_start('IDEA.NS', train), first)
            self.assertEqual(calls, [1])
        finally:
            finish.set()
            jobs._executor.shutdown()

    def test_failure_can_retry_and_completed_models_are_evicted(self):
        jobs = TrainingJobs()
        try:
            def fail(report):
                raise ValueError('failed')
            failed = jobs.get_or_start('x', fail)
            with self.assertRaises(ValueError):
                failed.future.result(2)
            retry = jobs.get_or_start('x', lambda report: 1)
            self.assertEqual(retry.future.result(2), 1)
            other = jobs.get_or_start('y', lambda report: 2)
            self.assertEqual(other.future.result(2), 2)
            self.assertNotIn('x', jobs._jobs)
        finally:
            jobs._executor.shutdown()


if __name__ == '__main__':
    unittest.main()
