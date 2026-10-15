# Experiment summary — ten simple points

1. **Two encoders were compared.** MobileFaceNet and ResNet50 ArcFace used the same aligned images within each comparison, making the recognition and attack results comparable under the stated protocols.
2. **Three protection methods were tested.** RO, SRHT, and PCA-guided RO produced 128-bit templates on LFW and 128-bit/512-bit templates on screened CFP.
3. **ResNet50 improved LFW recognition.** At 128 bits, its genuine acceptance was 92.28%, 92.10%, and 94.66%, compared with 82.99%, 84.48%, and 85.22% for MobileFaceNet, respectively.
4. **ResNet50 also improved CFP recognition.** In the matched screened-subset comparison, genuine acceptance rose from 37.76–43.50% to 65.12–66.94% at 128 bits, and from 81.95–82.52% to 94.37–94.67% at 512 bits.
5. **Renewal rejected most unchanged old records.** Changing the key usually stopped direct replay, but this did not remove the information already exposed through old templates.
6. **Template-only linkage showed a weak identity signal.** The tested learned scores were only modestly above chance; this does not prove that all possible linkage attacks would fail.
7. **Multiple exposures increased reconstruction risk.** Combining 1, 2, 5, and 10 old records with their known operators made the attack more effective, although the new operator remained hidden.
8. **Better recognition did not provide stronger protection.** With ten exposures, ResNet50 attack success on screened CFP reached 38.28–53.09% at 128 bits and 92.03–92.30% at 512 bits. Conditional paired intervals supported higher attack acceptance than MobileFaceNet in all six tested configurations.
9. **The templates save substantial storage.** A protected reference uses 16 or 64 bytes instead of 2048 bytes: 128-fold or 32-fold payload reductions. This is useful evidence for edge-oriented design, but device speed, memory, and energy were not measured.
10. **The paper's contributions are connected.** It shows the effect of repeated template exposure and explains how storage, recognition, and security must be considered together when choosing templates for kiosks or smart gadgets.

Recognition uses separately fitted training/validation thresholds targeting 0.1% FAR, with observed test FAR reported separately. LFW figures above are matcher-only results. CFP figures refer to the matched encoder replication; original baseline runs remain separate. Attack success means digital embedding-input acceptance against another image of the same person, not camera-level impersonation. This editorial revision changes presentation, not experimental evidence.
